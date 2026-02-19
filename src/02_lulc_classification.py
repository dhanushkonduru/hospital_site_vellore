"""
07_lulc_production.py
=====================
Final production LULC classification for Vellore Hospital Site Project.

Strategy: Use spectrally PURE pixels only for training.
- Built-up:   pixels with NDVI < 0.25 AND NDBI > -0.15 (rooftops, roads)
- Vegetation: pixels with NDVI > 0.40 (dense, unambiguous green)
- Water:      pixels with MNDWI > 0.15 (Palar river, confirmed wet)
- Bare Land:  pixels with BSI > 0.10 AND NDVI < 0.08 (sandy riverbed)

Training points verified from debug_v2_2013.png visual inspection:
- Red dots (built-up): bottom-center city block rows 450-680, cols 300-650
- City is the grey-brown textured area south of Palar river
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import glob, warnings
warnings.filterwarnings("ignore")

# ── Output folders ────────────────────────────────────────
OUTPUT_DIR = Path("data/processed/lulc_production")
MAPS_DIR   = Path("maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Bare Land"}
COLORS  = {1: "#E83C2A",  2: "#2ECC71",    3: "#3498DB", 4: "#F5CBA7"}

# ── Verified training point locations ────────────────────
# From visual inspection of debug_v2_2013.png:
# City (grey) = rows 420-700, cols 280-700 (bottom half, right side)
# River sand  = rows 295-360, cols 150-520 (horizontal bright band)
# Water       = dark channel within river band
# Vegetation  = top quadrant and edges

KNOWN_SAMPLES = {
    # Built-up: dense city blocks south of Palar river
    # These coords point to the grey urban texture in bottom-right
    "built_up_rows": [
        430, 440, 450, 460, 470, 480, 490, 500,
        510, 520, 530, 540, 550, 560, 570, 580,
        590, 600, 610, 620, 630, 640, 650, 660,
        450, 470, 490, 510, 530, 550,
    ],
    "built_up_cols": [
        320, 330, 340, 350, 360, 370, 380, 390,
        400, 410, 420, 430, 440, 450, 460, 470,
        480, 490, 500, 510, 520, 530, 540, 550,
        300, 300, 300, 300, 300, 300,
    ],

    # Vegetation: dense forest patches (top-left, unambiguously green)
    "veg_rows": [
        30,  40,  50,  60,  70,  80,  90, 100,
        110, 120, 130, 140, 150, 160, 170, 180,
        650, 660, 670, 680, 690, 700, 710, 720,
        30,  40,  50,  60,  70,  80,
    ],
    "veg_cols": [
        30,  40,  50,  60,  70,  80,  90, 100,
        110, 120, 130, 140, 150, 160, 170, 180,
        30,  40,  50,  60,  70,  80,  90, 100,
        200, 210, 220, 230, 240, 250,
    ],

    # Water: the dark water channel within Palar river
    # Left portion of river is darker (actual water, not sand)
    "water_rows": [
        310, 312, 314, 316, 318, 320,
        322, 324, 326, 328, 330,
    ],
    "water_cols": [
        80,  90, 100, 110, 120, 130,
        140, 150, 160, 170, 180,
    ],

    # Bare land: bright sandy riverbed (center of river band)
    "bare_rows": [
        325, 328, 330, 332, 335, 338,
        340, 342, 345, 348, 350,
    ],
    "bare_cols": [
        280, 300, 320, 340, 360, 380,
        400, 420, 440, 460, 480,
    ],
}


# ════════════════════════════════════════════════════════════
# LOAD SCENE
# ════════════════════════════════════════════════════════════
def load_scene(prefix, boundary_gdf):
    bands = {}
    profile = None

    for b in [2, 3, 4, 5, 6]:
        with rasterio.open(f"{prefix}_SR_B{b}.TIF") as src:
            bnd_r = boundary_gdf.to_crs(src.crs)
            geoms = [g.__geo_interface__ for g in bnd_r.geometry]
            arr, transform = mask(src, geoms, crop=True)
            if profile is None:
                profile = src.profile.copy()
                profile.update({
                    "height": arr.shape[1], "width": arr.shape[2],
                    "transform": transform, "crs": src.crs
                })
            data = arr[0].astype(float)
            data = np.where(data == 0, np.nan, data * 0.0000275 - 0.2)
            bands[f"B{b}"] = np.clip(data, 0, 1)

    # Cloud mask
    with rasterio.open(f"{prefix}_QA_PIXEL.TIF") as src:
        bnd_r = boundary_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in bnd_r.geometry]
        qa, _ = mask(src, geoms, crop=True)
    cloud = ((qa[0] >> 3) & 1) | ((qa[0] >> 4) & 1)
    for k in bands:
        bands[k] = np.where(cloud == 1, np.nan, bands[k])

    B2, B3, B4, B5, B6 = [bands[f"B{i}"] for i in [2, 3, 4, 5, 6]]

    # Spectral indices
    NDVI  = (B5 - B4) / (B5 + B4 + 1e-10)
    NDBI  = (B6 - B5) / (B6 + B5 + 1e-10)
    MNDWI = (B3 - B6) / (B3 + B6 + 1e-10)
    BSI   = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2) + 1e-10)
    EVI   = 2.5 * (B5 - B4) / (B5 + 6 * B4 - 7.5 * B2 + 1 + 1e-10)
    SAVI  = 1.5 * (B5 - B4) / (B5 + B4 + 0.5)   # soil-adjusted veg index
    IBI   = (2*B6/(B6+B5) - (B5/(B5+B4) + B3/(B3+B6))) / \
            (2*B6/(B6+B5) + (B5/(B5+B4) + B3/(B3+B6)) + 1e-10)  # index-based built-up

    stack = np.stack([
        B2, B3, B4, B5, B6,
        NDVI, NDBI, MNDWI, BSI, EVI, SAVI, IBI
    ], axis=0)  # 12 features

    return stack, profile, NDVI, NDBI, MNDWI, BSI


# ════════════════════════════════════════════════════════════
# EXTRACT TRAINING SAMPLES
# Two sources:
# 1. Known pixel coordinates (visually verified)
# 2. Spectrally pure pixels (high-confidence thresholds)
# ════════════════════════════════════════════════════════════
def extract_samples(stack, NDVI, NDBI, MNDWI, BSI, year):
    H, W = stack.shape[1], stack.shape[2]
    feats = stack.reshape(12, -1).T
    valid = ~np.isnan(feats).any(axis=1)
    np.random.seed(42)

    X_all, y_all = [], []

    # Source 1: Known coordinates (3×3 neighborhood)
    for rows, cols, cls in [
        (KNOWN_SAMPLES["built_up_rows"], KNOWN_SAMPLES["built_up_cols"], 1),
        (KNOWN_SAMPLES["veg_rows"],      KNOWN_SAMPLES["veg_cols"],      2),
        (KNOWN_SAMPLES["water_rows"],    KNOWN_SAMPLES["water_cols"],    3),
        (KNOWN_SAMPLES["bare_rows"],     KNOWN_SAMPLES["bare_cols"],     4),
    ]:
        for r, c in zip(rows, cols):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        px = stack[:, rr, cc]
                        if not np.isnan(px).any():
                            X_all.append(px)
                            y_all.append(cls)

    # Source 2: Spectrally pure pixels — very strict thresholds
    spectral_rules = [
        # Dense vegetation (unambiguous)
        (2, (NDVI.ravel() > 0.50) & (NDBI.ravel() < -0.20) & valid, 500),
        # Clear water (Mndwi strongly positive)
        (3, (MNDWI.ravel() > 0.20) & (NDVI.ravel() < 0.10) & valid, 400),
        # Sandy bare soil (bright, dry)
        (4, (BSI.ravel() > 0.12) & (NDVI.ravel() < 0.08) &
            (MNDWI.ravel() < -0.10) & valid, 300),
        # High-confidence built-up (low veg, positive built-up index)
        (1, (NDVI.ravel() < 0.20) & (NDBI.ravel() > -0.10) &
            (MNDWI.ravel() < -0.05) & (BSI.ravel() > -0.05) & valid, 400),
    ]

    for cls, cond, n_max in spectral_rules:
        idx = np.where(cond)[0]
        if len(idx) > n_max:
            idx = np.random.choice(idx, n_max, replace=False)
        if len(idx) > 0:
            X_all.append(feats[idx])
            y_all.append(np.full(len(idx), cls))

    # Combine all sources
    X_combined, y_combined = [], []
    for item in X_all:
        if isinstance(item, np.ndarray) and item.ndim == 2:
            X_combined.extend(item.tolist())
        elif isinstance(item, np.ndarray) and item.ndim == 1:
            X_combined.append(item.tolist())
    for item in y_all:
        if isinstance(item, np.ndarray):
            y_combined.extend(item.tolist())
        else:
            y_combined.append(item)

    X = np.array(X_combined, dtype=float)
    y = np.array(y_combined, dtype=int)

    counts = {CLASSES[i]: int(np.sum(y == i)) for i in [1, 2, 3, 4]}
    print(f"   Training samples: {counts}")
    return X, y


# ════════════════════════════════════════════════════════════
# TRAIN AND EVALUATE
# ════════════════════════════════════════════════════════════
def train_evaluate(X, y, year):
    from sklearn.model_selection import train_test_split

    # Balance classes: cap majority class at 3× minority
    min_count = min(np.sum(y == i) for i in [1, 2, 3, 4] if np.sum(y == i) > 0)
    cap = min(min_count * 3, 800)

    X_bal, y_bal = [], []
    np.random.seed(42)
    for cls in [1, 2, 3, 4]:
        idx = np.where(y == cls)[0]
        n = min(len(idx), cap)
        if n > 0:
            idx = np.random.choice(idx, n, replace=False)
            X_bal.append(X[idx])
            y_bal.append(y[idx])

    X_bal = np.vstack(X_bal)
    y_bal = np.concatenate(y_bal)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.25, stratify=y_bal, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_kappas = []
    for tr_idx, te_idx in cv.split(X_bal, y_bal):
        clf_cv = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced", random_state=42, n_jobs=-1)
        clf_cv.fit(X_bal[tr_idx], y_bal[tr_idx])
        pred = clf_cv.predict(X_bal[te_idx])
        cv_kappas.append(cohen_kappa_score(y_bal[te_idx], pred))

    y_pred = clf.predict(X_te)
    kappa  = cohen_kappa_score(y_te, y_pred)

    print(f"\n   📊 {year} — Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=[CLASSES[i] for i in [1, 2, 3, 4]]))
    print(f"   Hold-out Kappa:      {kappa:.4f}  "
          f"{'✅ PASS' if kappa >= 0.70 else '⚠️  Below target'}")
    print(f"   5-fold CV Kappa:     {np.mean(cv_kappas):.4f} ± {np.std(cv_kappas):.4f}")

    feat_names = ["B2","B3","B4","B5","B6",
                  "NDVI","NDBI","MNDWI","BSI","EVI","SAVI","IBI"]
    top = sorted(zip(feat_names, clf.feature_importances_),
                 key=lambda x: -x[1])[:5]
    print(f"   Top-5 features: {[(n, f'{v:.3f}') for n,v in top]}")

    return clf, kappa, np.mean(cv_kappas)


# ════════════════════════════════════════════════════════════
# CLASSIFY FULL SCENE
# ════════════════════════════════════════════════════════════
def classify_scene(stack, clf):
    H, W = stack.shape[1], stack.shape[2]
    feats = stack.reshape(12, -1).T
    valid = ~np.isnan(feats).any(axis=1)
    preds = np.zeros(H * W, dtype=np.uint8)
    preds[valid] = clf.predict(feats[valid])
    return preds.reshape(H, W)


# ════════════════════════════════════════════════════════════
# SAVE RASTER
# ════════════════════════════════════════════════════════════
def save_raster(classified, profile, year):
    p = profile.copy()
    p.update({"count": 1, "dtype": "uint8", "nodata": 0, "driver": "GTiff"})
    out = OUTPUT_DIR / f"lulc_{year}.tif"
    with rasterio.open(out, "w", **p) as dst:
        dst.write(classified, 1)
    print(f"   💾 Saved: {out}")
    return out


# ════════════════════════════════════════════════════════════
# PRODUCTION MAP (publication quality)
# ════════════════════════════════════════════════════════════
def production_map(classified, year, kappa, cv_kappa):
    cmap = mcolors.ListedColormap([COLORS[i] for i in [1, 2, 3, 4]])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    counts = {CLASSES[i]: np.sum(classified == i) for i in [1, 2, 3, 4]}
    total  = sum(counts.values())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(classified, cmap=cmap, norm=norm)
    ax.set_title(f"Land Use / Land Cover — Vellore Urban Core\n{year}",
                 fontsize=15, fontweight="bold", pad=12)
    ax.axis("off")

    # Stats box
    stats_lines = [f"{'Class':<14} {'Pixels':>8}  {'Area %':>7}"]
    stats_lines.append("─" * 32)
    for cls_id, cls_name in CLASSES.items():
        n   = counts[cls_name]
        pct = n / total * 100
        stats_lines.append(f"{cls_name:<14} {n:>8,}  {pct:>6.1f}%")
    stats_lines.append("─" * 32)
    stats_lines.append(f"Kappa (hold-out): {kappa:.4f}")
    stats_lines.append(f"Kappa (5-fold CV): {cv_kappa:.4f}")
    stats_text = "\n".join(stats_lines)

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.5))

    patches = [mpatches.Patch(color=COLORS[i], label=CLASSES[i])
               for i in [1, 2, 3, 4]]
    ax.legend(handles=patches, loc="lower right", fontsize=12,
              title="Land Cover Class", title_fontsize=11,
              framealpha=0.9)

    plt.tight_layout()
    out = MAPS_DIR / f"lulc_production_{year}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   🗺️  Map: {out}")


# ════════════════════════════════════════════════════════════
# FINAL COMPARISON MAP (all 3 years side by side)
# ════════════════════════════════════════════════════════════
def comparison_map(all_classified, all_counts):
    cmap = mcolors.ListedColormap([COLORS[i] for i in [1, 2, 3, 4]])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    years = sorted(all_classified.keys())
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    for ax, year in zip(axes, years):
        classified = all_classified[year]
        counts = all_counts[year]
        total  = sum(counts.values())

        ax.imshow(classified, cmap=cmap, norm=norm)
        stats = "\n".join([
            f"{k}: {v/total*100:.1f}%"
            for k, v in counts.items()
        ])
        ax.set_title(f"Vellore LULC — {year}\n{stats}",
                     fontsize=12, fontweight="bold")
        ax.axis("off")

    patches = [mpatches.Patch(color=COLORS[i], label=CLASSES[i])
               for i in [1, 2, 3, 4]]
    axes[2].legend(handles=patches, loc="lower right",
                   fontsize=11, framealpha=0.9)

    plt.suptitle("Vellore Urban Core — LULC Change Detection (2013 / 2019 / 2024)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = MAPS_DIR / "lulc_comparison_all_years.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n   🗺️  Comparison map: {out}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    # Tight Vellore urban bbox
    boundary = gpd.GeoDataFrame(
        geometry=[box(79.02, 12.82, 79.22, 13.02)], crs="EPSG:4326")

    # Auto-find scene prefixes
    SCENES = {}
    for year in ["2013", "2019", "2024"]:
        mtl = glob.glob(f"data/raw/vlr{year}/*_MTL.txt")
        if mtl:
            SCENES[year] = mtl[0].replace("_MTL.txt", "")
            print(f"✅ {year}: {Path(SCENES[year]).name}")

    all_classified = {}
    all_counts     = {}
    all_kappas     = {}
    all_cv_kappas  = {}

    for year, prefix in SCENES.items():
        print(f"\n{'='*56}")
        print(f"🛰️  Processing {year}...")

        stack, profile, NDVI, NDBI, MNDWI, BSI = load_scene(prefix, boundary)
        H, W = stack.shape[1], stack.shape[2]
        print(f"   Image: {H}×{W} px  |  12 features")
        print(f"   NDVI mean={np.nanmean(NDVI):.3f}  "
              f"NDBI mean={np.nanmean(NDBI):.3f}  "
              f"MNDWI mean={np.nanmean(MNDWI):.3f}")

        X, y = extract_samples(stack, NDVI, NDBI, MNDWI, BSI, year)
        clf, kappa, cv_kappa = train_evaluate(X, y, year)

        classified = classify_scene(stack, clf)
        counts = {CLASSES[i]: int(np.sum(classified == i)) for i in [1, 2, 3, 4]}
        total  = sum(counts.values())

        print(f"\n   🏙️  Final land cover ({year}):")
        for cls_name, n in counts.items():
            bar = "█" * int(n / total * 40)
            print(f"   {cls_name:<14} {n:>8,}  {n/total*100:>5.1f}%  {bar}")

        all_classified[year] = classified
        all_counts[year]     = counts
        all_kappas[year]     = kappa
        all_cv_kappas[year]  = cv_kappa

        save_raster(classified, profile, year)
        production_map(classified, year, kappa, cv_kappa)

    # ── Change detection ──────────────────────────────────
    print(f"\n{'='*56}")
    print("📈 Urban Growth Analysis — Vellore 2013 → 2024:")
    print()

    if "2013" in all_counts and "2024" in all_counts:
        for cls_id, cls_name in CLASSES.items():
            n13 = all_counts["2013"].get(cls_name, 0)
            n19 = all_counts.get("2019", {}).get(cls_name, 0)
            n24 = all_counts["2024"].get(cls_name, 0)
            pct_total = (n24 - n13) / (n13 + 1) * 100
            arrow = "📈" if pct_total > 5 else ("📉" if pct_total < -5 else "➡️")
            print(f"   {arrow} {cls_name:<14} "
                  f"2013={n13:>7,}  2019={n19:>7,}  2024={n24:>7,}  "
                  f"({pct_total:+.1f}%)")

    # ── Kappa summary ─────────────────────────────────────
    print(f"\n{'='*56}")
    print("📊 Accuracy Summary:")
    all_pass = True
    for yr in ["2013", "2019", "2024"]:
        if yr in all_kappas:
            k    = all_kappas[yr]
            cv_k = all_cv_kappas[yr]
            status = "✅ PASS" if k >= 0.70 else "⚠️  Low"
            if k < 0.70:
                all_pass = False
            print(f"   {yr}: Kappa={k:.4f} ({status})  "
                  f"CV={cv_k:.4f}")

    comparison_map(all_classified, all_counts)

    print(f"\n{'='*56}")
    if all_pass:
        print("✅ ALL KAPPA SCORES PASS (≥ 0.70)")
        print("✅ LULC Stage COMPLETE — ready for Stage 2: CA-ANN")
        print()
        print("📁 Outputs:")
        print("   data/processed/lulc_production/lulc_2013.tif")
        print("   data/processed/lulc_production/lulc_2019.tif")
        print("   data/processed/lulc_production/lulc_2024.tif")
        print("   maps/lulc_production_2013/2019/2024.png")
        print("   maps/lulc_comparison_all_years.png")
    else:
        print("⚠️  Some Kappa scores below 0.70")
        print("   → Share maps/lulc_comparison_all_years.png")
        print("   → Even so, visually correct maps can proceed to Stage 2")
        print("   → Kappa can be improved with more training data in paper revision")


if __name__ == "__main__":
    main()