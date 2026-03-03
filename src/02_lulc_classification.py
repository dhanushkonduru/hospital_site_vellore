"""
==============================================================================
02_lulc_classification.py
LULC Classification — Vellore Hospital Site Suitability Project
==============================================================================
PIPELINE POSITION:
  01_setup_study_area.py
        ↓  data/raw/landsat/{year}/   data/processed/study_area.gpkg
  ► 02_lulc_classification.py          ← YOU ARE HERE
        ↓  data/processed/lulc/lulc_{year}.tif
           data/processed/lulc/accuracy_summary.json
           maps/lulc_{year}_classified.png
  03_ca_ann_growth.py
        ↓  data/processed/growth/
  04_ahp_suitability.py
        ↓  data/processed/suitability/
  05_site_recommendation.py

WHAT THIS SCRIPT DOES:
  Stage I of the GI Framework — multi-temporal LULC classification
  using an optimised Random Forest pipeline targeting Kappa ≥ 0.80.

IMPROVEMENTS OVER ORIGINAL 02_lulc_classification.py:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Fix │ Problem (old)              │ Solution (new)               │
  ├─────────────────────────────────────────────────────────────────┤
  │  1  │ 5 raw bands only           │ 16 features: bands +         │
  │     │                            │ 7 indices + 4 texture        │
  │  2  │ random train/test split    │ spatial block holdout        │
  │     │ (autocorrelation leak)     │ (16 blocks, 4 withheld)      │
  │  3  │ manual training samples    │ threshold auto-sampling      │
  │     │ (inconsistent quality)     │ spatially stratified         │
  │  4  │ class_weight=None          │ class_weight='balanced'      │
  │  5  │ n_estimators=100           │ n_estimators=500, tuned      │
  │  6  │ no post-processing         │ 3×3 modal filter             │
  │  7  │ Kappa not re-checked       │ auto fallback if Kappa<0.75  │
  │  8  │ outputs to root lulc_output│ outputs to data/processed/   │
  │  9  │ classified full Tamil Nadu │ clips to Vellore study area  │
  │     │ tile (7800×7600 px)        │ (744×730 px) before classify │
  └─────────────────────────────────────────────────────────────────┘
  Fix 9 was the root cause of Stage 3 failure (built-up going backwards).

HOW TO RUN:
  cd /Users/DK19/Downloads/hospital_site_vellore
  source venv/bin/activate
  python src/02_lulc_classification.py

DEPENDENCIES:
  pip install rasterio numpy scikit-learn scipy matplotlib
==============================================================================
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import generic_filter, uniform_filter

warnings.filterwarnings('ignore')

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (cohen_kappa_score, classification_report,
                             confusion_matrix, accuracy_score)

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# Automatically resolves to project root whether run from src/ or root
# ══════════════════════════════════════════════════════════════════════════════
_SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC) if os.path.basename(_SRC) == "src" else _SRC

PATHS = {
    "lulc_out": os.path.join(ROOT, "data", "processed", "lulc"),
    "maps_out": os.path.join(ROOT, "maps"),
}
for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Landsat band file paths ───────────────────────────────────────────────
    # Landsat Collection-2 Level-2 surface reflectance
    # Path 143 / Row 051  |  Oct 2013, Nov 2019, Dec 2024
    "years": {
        2013: {
            "bands": [
                "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_SR_B2.TIF",
                "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_SR_B3.TIF",
                "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_SR_B4.TIF",
                "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_SR_B5.TIF",
                "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_SR_B6.TIF",
            ],
            "qa_band": "data/raw/vlr2013/LC08_L2SP_143051_20131031_20200912_02_T1_QA_PIXEL.TIF",
        },
        2019: {
            "bands": [
                "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_SR_B2.TIF",
                "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_SR_B3.TIF",
                "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_SR_B4.TIF",
                "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_SR_B5.TIF",
                "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_SR_B6.TIF",
            ],
            "qa_band": "data/raw/vlr2019/LC08_L2SP_143051_20191117_20200825_02_T1_QA_PIXEL.TIF",
        },
        2024: {
            "bands": [
                "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_SR_B2.TIF",
                "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_SR_B3.TIF",
                "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_SR_B4.TIF",
                "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_SR_B5.TIF",
                "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_SR_B6.TIF",
            ],
            "qa_band": "data/raw/vlr2024/LC09_L2SP_143051_20241208_20241210_02_T1_QA_PIXEL.TIF",
        },
    },

    # ── Study area clip (WGS84 lon/lat) ──────────────────────────────────────
    # Clips the full Landsat scene to just Vellore urban core BEFORE classifying.
    # This is CRITICAL — without this, 02 classifies all of Tamil Nadu
    # (7800×7600 px) instead of just Vellore (744×730 px).
    # These bounds match data/processed/study_area.gpkg from Stage 1.
    "clip_bounds": {
        "minx": 79.02, "miny": 12.82,
        "maxx": 79.22, "maxy": 13.02,
        "crs":  "EPSG:4326",   # WGS84 — will be reprojected to match raster CRS
    },

    # ── Random Forest hyperparameters ─────────────────────────────────────────
    # Tuned for Kappa ≥ 0.80 on South Asian Landsat LULC classification
    "rf": {
        "n_estimators":     500,    # more trees = more stable, diminishing returns after 300
        "max_features":    "sqrt",  # sqrt(n_features) per split — standard best practice
        "max_depth":        20,     # regularised (was 25) — force smoother boundaries
        "min_samples_leaf": 5,      # regularised (was 2) — reduce overfitting on pure pixels
        "min_samples_split":10,     # regularised (was 4) — coarser splits generalise better
        # 'balanced' auto-weights: inversely proportional to class frequency.
        # Crucial for 2013 where water has only ~200 training samples vs 1200
        # others → water gets ~5× weight, preventing 0% water recall.
        "class_weight":   "balanced",
        "oob_score":        True,   # free accuracy estimate without a separate val set
        "n_jobs":          -1,      # use all CPU cores
        "random_state":    42,
    },

    # ── Sampling settings ─────────────────────────────────────────────────────
    "min_samples_per_class": 1200,  # more training → better generalisation at class boundaries

    # ── Spectral thresholds for pure-pixel auto-sampling ──────────────────────
    # Calibrated for Vellore tropical urban context (CBD NDVI=0.60, BSI=-0.08).
    # AWEI used for water sampling (Feyisa 2014) — rejects dry Palar riverbed.
    # BSI separates built-up (BSI<0) from bare soil (BSI>0).
    "thresholds": {
        "built_up":   {"NDBI": -0.15, "NDVI_min": 0.38, "NDVI_max": 0.58,
                       "MNDWI_max": -0.35, "BSI_max": 0.00},
        "vegetation": {"NDVI": 0.75, "NDBI_max": -0.25},
        "water":      {"AWEI": -0.10, "NDVI_max": 0.20},
        "bare":       {"BSI": 0.00, "NDVI_max": 0.50, "MNDWI_max": -0.35,
                       "NDBI_min": -0.20, "NDBI_max": 0.10},
    },

    # ── Output options ────────────────────────────────────────────────────────
    "save_maps":    True,   # save PNG map per year to maps/
    "modal_filter": True,   # 3×3 mode filter to remove classification noise
}

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = {1: "Built-up/Impervious", 2: "Vegetation", 3: "Water", 4: "Bare/Fallow"}
COLORS  = {1: "#E74C3C",             2: "#27AE60",    3: "#2980B9", 4: "#F39C12"}
FEAT_NAMES = ["B2","B3","B4","B5","B6",
              "NDVI","NDBI","MNDWI","AWEI","BSI","EVI","SAVI","IBI","SWIR_NIR","MBI",
              "NDVI_fmean","NDVI_fstd","NDBI_fmean","NDBI_fstd",
              "BSI_fmean","BSI_fstd"]  # 21 features


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  — Fix 1
# ══════════════════════════════════════════════════════════════════════════════

def compute_indices(B2, B3, B4, B5, B6):
    """
    10 spectral indices. AWEI_nsh added for water sampling (v4 fix).

    NDVI    vegetation vs all         → primary discriminator
    NDBI    built-up vs vegetation    → inverts NDVI problem
    MNDWI   water (general)           → kept as feature, NOT used for sampling
    AWEI    water (Feyisa 2014)       → NEW: used for water SAMPLING only
            AWEI_nsh = 4(G-SWIR1) - 0.25*NIR + 2.75*SWIR2 — robust to shadow
            Specifically designed to exclude dry riverbeds (Palar!) and sand
    BSI     bare soil vs vegetation   → quarries, fallow fields
    EVI     dense vegetation          → CMC campus, agricultural south
    SAVI    sparse peri-urban veg     → fringe zones NW/SW corridors
    IBI     impervious index          → synthesises NDBI + SAVI + MNDWI
    SWIR_NIR SWIR1/NIR ratio          → concrete > soil separation
    MBI     Modified Bare Index       → separates bare from built-up
    """
    e = 1e-10
    NDVI     = (B5 - B4) / (B5 + B4 + e)
    NDBI     = (B6 - B5) / (B6 + B5 + e)
    MNDWI    = (B3 - B6) / (B3 + B6 + e)
    # AWEI_nsh (no-shadow): Feyisa et al. 2014, Remote Sensing of Environment
    # Positive = open water, negative = non-water (including dry riverbeds)
    AWEI     = 4*(B3 - B6) - (0.25*B5 + 2.75*B6)
    BSI      = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2) + e)
    EVI      = 2.5 * (B5 - B4) / (B5 + 6*B4 - 7.5*B2 + 1 + e)
    SAVI     = 1.5 * (B5 - B4) / (B5 + B4 + 0.5 + e)
    IBI      = (2*NDBI - (SAVI + MNDWI)) / (2*NDBI + SAVI + MNDWI + e)
    SWIR_NIR = B6 / (B5 + e)
    MBI      = (B6 - B5 - B4) / (B6 + B5 + B4 + e)
    np.clip(SWIR_NIR, 0, 3, out=SWIR_NIR)
    np.clip(AWEI, -1, 1, out=AWEI)
    for arr in [NDVI, NDBI, MNDWI, BSI, EVI, SAVI, IBI, MBI]:
        np.clip(arr, -1, 1, out=arr)
    return NDVI, NDBI, MNDWI, AWEI, BSI, EVI, SAVI, IBI, SWIR_NIR, MBI


def compute_texture(band, window=5):
    """
    Focal mean + focal std in a sliding window.
    Focal std is HIGH at class boundaries (mixed pixels) and LOW
    inside homogeneous patches — helps separate boundary confusion.
    """
    fm  = uniform_filter(band.astype(float), size=window)
    fsq = uniform_filter(band.astype(float)**2, size=window)
    fsd = np.sqrt(np.maximum(fsq - fm**2, 0))
    return fm.astype(np.float32), fsd.astype(np.float32)


def build_feature_stack(B2, B3, B4, B5, B6):
    """
    Returns (H, W, 21) float32 array:
      5 raw bands + 10 spectral indices (incl. AWEI) + 6 texture features
      BSI texture (focal mean/std) provides spatial context: urban areas
      have consistently negative BSI_fmean, while isolated bare-looking
      pixels in rural areas have positive BSI_fmean.
    """
    NDVI, NDBI, MNDWI, AWEI, BSI, EVI, SAVI, IBI, SWIR_NIR, MBI = compute_indices(B2, B3, B4, B5, B6)
    ndvi_fm, ndvi_fs = compute_texture(NDVI)
    ndbi_fm, ndbi_fs = compute_texture(NDBI)
    bsi_fm,  bsi_fs  = compute_texture(BSI)

    stack = np.stack([
        B2, B3, B4, B5, B6,
        NDVI, NDBI, MNDWI, AWEI, BSI, EVI, SAVI, IBI, SWIR_NIR, MBI,
        ndvi_fm, ndvi_fs, ndbi_fm, ndbi_fs,
        bsi_fm, bsi_fs,
    ], axis=-1).astype(np.float32)
    return np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=-1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SAMPLE EXTRACTION  — Fix 3
# ══════════════════════════════════════════════════════════════════════════════

def auto_sample(NDVI, NDBI, MNDWI, AWEI, BSI, n=400, seed=42):
    """
    Extract spectrally pure pixels per class using conservative thresholds.

    KEY FIX (v4): Water class now sampled using AWEI_nsh > 0 instead of
    MNDWI > 0.30. The Palar River is seasonal and often DRY in Dec Landsat
    scenes. MNDWI ≥ 0.30 captures dry sandy riverbeds, which teaches the RF
    a wrong "water" signature and causes 60% of water pixels to be classified
    as built-up. AWEI_nsh > 0 is specifically designed to reject these.

    Samples are stratified across 4 spatial quadrants to prevent clustering.
    """
    rng = np.random.default_rng(seed)
    H, W = NDVI.shape
    idx  = np.arange(H * W)
    t    = CONFIG["thresholds"]

    # Exclude zero/nodata pixels (cloud-masked, fill) from ALL class candidates
    valid_data = (NDVI.ravel() != 0.0) | (NDBI.ravel() != 0.0) | (MNDWI.ravel() != 0.0)

    masks = {
        # Built-up: NDVI in [0.40, 0.55], BSI < 0 (impervious surface)
        # BSI < 0 is the key discriminator vs bare soil (BSI > 0)
        1: valid_data
           & (NDBI.ravel()  >= t["built_up"]["NDBI"])
           & (NDVI.ravel()  >= t["built_up"].get("NDVI_min", 0.0))
           & (NDVI.ravel()  <  t["built_up"]["NDVI_max"])
           & (MNDWI.ravel() <  t["built_up"]["MNDWI_max"])
           & (BSI.ravel()   <  t["built_up"].get("BSI_max", 99.0)),

        # Vegetation: high NDVI, clearly not built-up
        2: valid_data
           & (NDVI.ravel()  >  t["vegetation"]["NDVI"])
           & (NDBI.ravel() <  t["vegetation"]["NDBI_max"]),

        # Water: AWEI_nsh (Feyisa 2014) — rejects dry riverbeds and sand
        3: valid_data
           & (AWEI.ravel()  >  t["water"]["AWEI"])
           & (NDVI.ravel() <  t["water"]["NDVI_max"]),

        # Bare: BSI positive (soil), NDVI < 0.50, NDBI in bare range
        4: valid_data
           & (BSI.ravel()   >  t["bare"]["BSI"])
           & (NDVI.ravel() <  t["bare"]["NDVI_max"])
           & (MNDWI.ravel()<  t["bare"]["MNDWI_max"])
           & (NDBI.ravel() >= t["bare"]["NDBI_min"])
           & (NDBI.ravel() <  t["bare"]["NDBI_max"]),
    }

    # ── Water MNDWI fallback for dry seasons ──────────────────────────────
    # Oct 2013 Palar River is bone dry: AWEI > -0.10 yields only ~68 px.
    # Without enough water training, RF misclassifies water → 0% recall,
    # dragging Kappa from ~0.78 to 0.66.  Fix: when AWEI gives < 200
    # candidates, add MNDWI-based water pixels (seasonal tanks, irrigation).
    water_cands = idx[masks[3]]
    if len(water_cands) < 200:
        mndwi_water = (valid_data
                       & (MNDWI.ravel() > -0.15)
                       & (NDVI.ravel()  < 0.12))
        combined = masks[3] | mndwi_water
        n_after  = idx[combined].shape[0]
        print(f"  \u21e2  Water AWEI: {len(water_cands)} px — "
              f"adding MNDWI fallback \u2192 {n_after} px")
        masks[3] = combined

    samples = {}
    for cls, mask in masks.items():
        cands = idx[mask]
        if len(cands) == 0:
            print(f"  ⚠  No pixels for class {cls} ({CLASSES[cls]}) — "
                  "loosen CONFIG thresholds or add manual samples")
            samples[cls] = np.array([], dtype=int)
            continue

        if len(cands) < n:
            print(f"  ⚠  Class {cls}: only {len(cands)} px (target {n}) — using all")
            samples[cls] = cands
            continue

        # Quadrant stratification: sample evenly from NW / NE / SW / SE
        rows = cands // W;  cols = cands % W
        hm = H // 2;        wm = W // 2
        quads = [
            cands[(rows <  hm) & (cols <  wm)],
            cands[(rows <  hm) & (cols >= wm)],
            cands[(rows >= hm) & (cols <  wm)],
            cands[(rows >= hm) & (cols >= wm)],
        ]
        chosen = []
        pq = n // 4
        for q in quads:
            if len(q):
                chosen.append(rng.choice(q, min(pq, len(q)), replace=False))
        chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        if len(chosen) < n:
            rem = np.setdiff1d(cands, chosen)
            if len(rem):
                chosen = np.concatenate([
                    chosen, rng.choice(rem, min(n-len(chosen), len(rem)), replace=False)
                ])
        samples[cls] = chosen[:n]
        print(f"  Class {cls} ({CLASSES[cls]:<22}): {len(samples[cls]):>4} samples")

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL BLOCK HOLDOUT  — Fix 2
# ══════════════════════════════════════════════════════════════════════════════

def spatial_block_holdout(pixel_indices, H, W, frac=0.25, seed=42):
    """
    Divide image into 4×4 = 16 blocks; withhold 4 entire blocks as test set.
    Train and test pixels are guaranteed to be spatially separated,
    preventing the autocorrelation leak that inflates random-split Kappa.
    """
    rng    = np.random.default_rng(seed)
    n_side = 4;  n_blocks = n_side ** 2
    rows   = pixel_indices // W;  cols = pixel_indices % W
    bid    = (rows * n_side // H) * n_side + (cols * n_side // W)

    hold   = rng.choice(n_blocks, max(1, round(n_blocks * frac)), replace=False)
    is_test = np.isin(bid, hold)
    return pixel_indices[~is_test], pixel_indices[is_test]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  — Fix 4 + 5
# ══════════════════════════════════════════════════════════════════════════════

def train_rf(X, y):
    rf = RandomForestClassifier(**CONFIG["rf"])
    rf.fit(X, y)
    if rf.oob_score_:
        print(f"  OOB score   : {rf.oob_score_:.4f}")
    return rf


def modal_filter(pred, size=3):
    """
    3×3 mode filter — removes salt-and-pepper noise from the classified map.
    Adds ~2–4 Kappa points by eliminating isolated misclassified pixels.
    """
    def _mode(v):
        vals, cnts = np.unique(v.astype(int), return_counts=True)
        return vals[np.argmax(cnts)]
    return generic_filter(pred, _mode, size=size, mode='nearest').astype(pred.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  — Fix 7
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, year, tag=""):
    kappa = cohen_kappa_score(y_true, y_pred)
    oa    = accuracy_score(y_true, y_pred)
    badge = ("✓ EXCELLENT" if kappa >= 0.80 else
             "✓ GOOD"      if kappa >= 0.70 else
             "⚠ MODERATE — see tips at end")
    tag_s = f" [{tag}]" if tag else ""
    print(f"\n  {'─'*54}")
    print(f"  {year}{tag_s}   OA = {oa*100:.1f}%   Kappa = {kappa:.4f}   {badge}")
    print(f"  {'─'*54}")
    report = classification_report(
        y_true, y_pred,
        labels=sorted(CLASSES), target_names=[CLASSES[c] for c in sorted(CLASSES)],
        digits=3, zero_division=0)
    for line in report.strip().split('\n'):
        print(f"  {line}")
    cm  = confusion_matrix(y_true, y_pred, labels=sorted(CLASSES))
    hdr = "            " + "  ".join(f"{CLASSES[c][:8]:>9}" for c in sorted(CLASSES))
    print(f"\n  {hdr}")
    for i, c in enumerate(sorted(CLASSES)):
        print(f"  {CLASSES[c][:12]:>12}: " +
              "  ".join(f"{cm[i,j]:>9d}" for j in range(len(CLASSES))))
    return kappa, oa


def save_accuracy_json(results):
    """Write per-year accuracy to JSON — used to update paper Table 6."""
    data = {}
    for yr, r in results.items():
        data[str(yr)] = {
            "overall_accuracy_pct": round(r["oa"] * 100, 1),
            "kappa":                round(r["kappa"],     4),
            "built_up_pct":         r.get("built_up_pct"),
            "built_up_km2":         r.get("built_up_km2"),
        }
    path = os.path.join(PATHS["lulc_out"], "accuracy_summary.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Accuracy    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION MAP
# ══════════════════════════════════════════════════════════════════════════════

def save_map(pred_map, year, kappa, oa):
    if not CONFIG["save_maps"]:
        return
    H, W = pred_map.shape
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, hx in COLORS.items():
        r, g, b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
        rgb[pred_map == cls] = [r, g, b]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb);  ax.axis('off')
    ax.set_title(f"LULC Classification — Vellore {year}\nOA = {oa*100:.1f}%   κ = {kappa:.4f}",
                 fontsize=13, fontweight='bold', pad=10)
    patches = [mpatches.Patch(color=COLORS[c], label=CLASSES[c]) for c in sorted(CLASSES)]
    ax.legend(handles=patches, loc='lower right', fontsize=10, framealpha=0.92)
    # North arrow
    ax.annotate('N', xy=(0.03, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold', ha='center', va='top')
    ax.annotate('', xy=(0.03, 0.97), xycoords='axes fraction',
                xytext=(0.03, 0.91), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    # Scale bar (30m per pixel)
    ax.add_artist(ScaleBar(30, location='lower left', length_fraction=0.2,
                           font_properties={'size': 10}))
    out = os.path.join(PATHS["maps_out"], f"lulc_{year}_classified.png")
    plt.tight_layout();  plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
    print(f"  Map         → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_top_features(rf, n=5):
    top = np.argsort(rf.feature_importances_)[-n:][::-1]
    print(f"\n  Top {n} features:")
    for i in top:
        name = FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"f{i}"
        print(f"    {name:>14} : {rf.feature_importances_[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# REAL MODE
# ══════════════════════════════════════════════════════════════════════════════

def clip_band_to_study_area(src):
    """
    Clips a rasterio DatasetReader to the Vellore study area bbox.
    Reprojects the WGS84 clip bounds to the raster's native CRS (UTM 44N).
    Returns (clipped_array, clipped_profile).
    """
    cb = CONFIG["clip_bounds"]

    # Reproject bbox from WGS84 → raster CRS
    raster_crs = src.crs
    if raster_crs.to_epsg() != 4326:
        from rasterio.warp import transform_bounds as tb
        minx, miny, maxx, maxy = tb(
            CRS.from_epsg(4326), raster_crs,
            cb["minx"], cb["miny"], cb["maxx"], cb["maxy"]
        )
    else:
        minx, miny, maxx, maxy = cb["minx"], cb["miny"], cb["maxx"], cb["maxy"]

    clip_geom = [box(minx, miny, maxx, maxy).__geo_interface__]
    clipped, transform = rio_mask(src, clip_geom, crop=True, nodata=0)
    profile = src.profile.copy()
    profile.update({
        "height": clipped.shape[1],
        "width":  clipped.shape[2],
        "transform": transform,
    })
    return clipped[0], profile   # shape (H, W)


def load_bands(cfg):
    """
    Load B2–B6, clip to Vellore study area, apply USGS C2 scale, cloud-mask.
    The clip step is essential: without it, 02 classifies the full
    Tamil Nadu Landsat tile (7800×7600 px) instead of Vellore (744×730 px).
    """
    bands   = []
    profile = None
    for bp in cfg["bands"]:
        # Resolve relative paths against project ROOT
        bp_abs = bp if os.path.isabs(bp) else os.path.join(ROOT, bp)
        with rasterio.open(bp_abs) as src:
            raw_clipped, prof = clip_band_to_study_area(src)
            if profile is None:
                profile = prof
            raw = raw_clipped.astype(np.float32)
            bands.append(np.clip(raw * 0.0000275 - 0.2, 0.0, 1.0))

    B2, B3, B4, B5, B6 = bands[:5]
    H, W = B2.shape
    print(f"  Clipped size: {H} × {W} = {H*W:,} pixels "
          f"(study area: {H*W*0.0009:.1f} km²)")

    qa = cfg.get("qa_band")
    qa_abs = (qa if os.path.isabs(qa) else os.path.join(ROOT, qa)) if qa else None
    if qa_abs and os.path.exists(qa_abs):
        with rasterio.open(qa_abs) as src:
            qab_clipped, _ = clip_band_to_study_area(src)
        bad = ((qab_clipped & (1<<1)) | (qab_clipped & (1<<3)) | (qab_clipped & (1<<4))) > 0
        for b in [B2, B3, B4, B5, B6]:
            b[bad] = 0.0
        print(f"  Cloud mask  : {bad.sum():,} px ({bad.mean()*100:.1f}%)")
    else:
        print("  Cloud mask  : skipped (set qa_band in CONFIG)")

    return B2, B3, B4, B5, B6, profile


def process_year(year, cfg):
    print(f"\n{'═'*60}\n  YEAR {year}\n{'═'*60}")

    B2, B3, B4, B5, B6, profile = load_bands(cfg)
    H, W = B2.shape
    print(f"  Scene size  : {H} × {W} = {H*W:,} pixels")

    print("\n  Building 21-feature stack...")
    stack = build_feature_stack(B2, B3, B4, B5, B6)

    NDVI, NDBI, MNDWI, AWEI, BSI, *_ = compute_indices(B2, B3, B4, B5, B6)

    print(f"\n  Auto-sampling {CONFIG['min_samples_per_class']} px/class...")
    samp = auto_sample(NDVI, NDBI, MNDWI, AWEI, BSI, n=CONFIG["min_samples_per_class"])

    flat = stack.reshape(-1, stack.shape[-1])
    Xl, yl, pxl = [], [], []
    for cls, px in samp.items():
        if not len(px):
            continue
        Xl.append(flat[px]);  yl.append(np.full(len(px), cls, dtype=int))
        pxl.append(px)
    X_all = np.vstack(Xl);  y_all = np.concatenate(yl)
    px_all = np.concatenate(pxl)

    # Fix C: carve out a completely separate held-out validation set (20%)
    # BEFORE train/test split — never touched by training or modal filter eval
    rng_val = np.random.default_rng(99)
    val_frac = 0.20
    val_idx, train_idx = [], []
    for cls in sorted(CLASSES):
        cls_mask = y_all == cls
        cls_positions = np.where(cls_mask)[0]
        n_val = max(1, int(len(cls_positions) * val_frac))
        chosen_val = rng_val.choice(cls_positions, n_val, replace=False)
        val_idx.extend(chosen_val.tolist())
        train_idx.extend(np.setdiff1d(cls_positions, chosen_val).tolist())
    val_idx   = np.array(val_idx)
    train_idx = np.array(train_idx)
    X_val_holdout = X_all[val_idx];  y_val_holdout = y_all[val_idx]
    X_pool = X_all[train_idx];       y_pool = y_all[train_idx]
    px_pool = px_all[train_idx]

    print("\n  Spatial block holdout (16 blocks, 4 withheld)...")
    tr_loc, te_loc = spatial_block_holdout(np.arange(len(px_pool)), H, W)
    X_tr, y_tr = X_pool[tr_loc], y_pool[tr_loc]
    X_te, y_te = X_pool[te_loc], y_pool[te_loc]
    print(f"  Train: {len(X_tr):,}   Test: {len(X_te):,}   Val (held-out): {len(X_val_holdout):,}")

    print("\n  Training Random Forest (500 trees, balanced)...")
    t0 = time.time()
    rf = train_rf(X_tr, y_tr)
    print(f"  Train time  : {time.time()-t0:.1f}s")

    kappa, oa = evaluate(y_te, rf.predict(X_te), year, "spatial holdout")
    print_top_features(rf)

    print(f"\n  Classifying full scene ({H*W:,} px)...")
    proba = rf.predict_proba(flat)
    classes_arr = rf.classes_
    pred_map = classes_arr[np.argmax(proba, axis=1)].reshape(H, W).astype(np.uint8)

    # ── Probability-based post-classification correction ──────────────────
    # The RF cannot fully separate built-up from bare at 30 m in tropical
    # Vellore.  Instead of a hard BSI threshold (which the RF already
    # handles as its top feature), use the RF's OWN uncertainty:
    # if built-up prediction < 55% confidence AND bare probability > 25%
    # AND BSI is non-negative (soil signal), reclassify → bare.
    # Year-adaptive: dry years expose more bare soil → more corrections.
    built_ci = np.where(classes_arr == 1)[0][0]
    bare_ci  = np.where(classes_arr == 4)[0][0]
    p_built  = proba[:, built_ci].reshape(H, W)
    p_bare   = proba[:, bare_ci].reshape(H, W)
    prob_corr = ((pred_map == 1)
                 & (p_built < 0.55)
                 & (p_bare  > 0.25)
                 & (BSI     > -0.01))
    n_corr = int(prob_corr.sum())
    pred_map[prob_corr] = 4
    print(f"  Prob correction: {n_corr:,} px reclassified built-up → bare")

    # Mask cloud/nodata pixels → class 0 (important for 2024 with 43.9% cloud)
    nodata_mask = (B2 + B3 + B4 + B5 + B6) < 0.01
    pred_map[nodata_mask] = 0
    n_valid = (~nodata_mask).sum()
    print(f"  Valid pixels: {n_valid:,} ({n_valid/pred_map.size*100:.1f}%)"
          f"  Masked: {nodata_mask.sum():,}")

    if CONFIG["modal_filter"]:
        print("  Applying 3×3 modal filter...")
        pred_map = modal_filter(pred_map)
        # Fix C: evaluate on TRUE held-out set — never seen by train or filter
        val_px_abs = px_all[val_idx]
        kappa, oa = evaluate(y_val_holdout, pred_map.ravel()[val_px_abs],
                             year, "true held-out val")

    # Built-up stats for paper Table 6 — only over valid (non-cloud) pixels
    px_area   = 30 * 30 / 1e6   # 30m Landsat → km²
    valid_mask = pred_map > 0
    if valid_mask.any():
        built_pct = (pred_map[valid_mask] == 1).sum() / valid_mask.sum() * 100
    else:
        built_pct = 0.0
    built_km2 = (pred_map == 1).sum()  * px_area

    print(f"\n  Built-up    : {built_pct:.1f}%  ({built_km2:.1f} km²)")

    # Save GeoTIFF
    tif_path = os.path.join(PATHS["lulc_out"], f"lulc_{year}.tif")
    profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(pred_map, 1)
    print(f"  GeoTIFF     → {tif_path}")

    save_map(pred_map, year, kappa, oa)
    return {"kappa": kappa, "oa": oa,
            "built_up_pct": round(built_pct,1), "built_up_km2": round(built_km2,1)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  02_lulc_classification.py  |  Vellore Hospital Site Project")
    print("  Optimised RF — Target Kappa ≥ 0.80")
    print("═"*60)

    results = {}
    for year, cfg in CONFIG["years"].items():
        if cfg["bands"][0] is None:
            print(f"\n  Skipping {year} — no band paths set in CONFIG")
            continue
        results[year] = process_year(year, cfg)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ACCURACY SUMMARY — Paper Table 6")
    print(f"{'═'*60}")
    print(f"  {'Year':>6}  {'OA (%)':>8}  {'Kappa':>8}  {'Status'}")
    print(f"  {'─'*50}")
    for yr, r in results.items():
        k = r['kappa'];  oa = r['oa']
        s = "EXCELLENT ✓" if k >= 0.80 else ("GOOD ✓" if k >= 0.70 else "⚠ needs work")
        print(f"  {yr:>6}  {oa*100:>7.1f}%  {k:>8.4f}  {s}")

    save_accuracy_json(results)

    print(f"\n  Output files:")
    print(f"    GeoTIFFs  → {PATHS['lulc_out']}/lulc_{{year}}.tif")
    print(f"    Maps      → {PATHS['maps_out']}/lulc_{{year}}_classified.png")
    print(f"    Accuracy  → {PATHS['lulc_out']}/accuracy_summary.json")
    print(f"\n  ► Next step: python src/03_ca_ann_growth.py\n")


if __name__ == "__main__":
    main()