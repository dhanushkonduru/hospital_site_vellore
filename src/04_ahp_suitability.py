"""
04_ahp_suitability.py
=====================
Stage 3: AHP Multi-Criteria Suitability Analysis
Hospital Site Suitability — Vellore, Tamil Nadu

Pipeline:
1. Compute 6 criteria rasters from existing project data
2. Normalise all criteria to 0-1
3. AHP pairwise matrix → eigenvector weights → verify CR < 0.10
4. Weighted overlay → suitability raster
5. Classify into High / Medium / Low
6. Save rasters + production maps
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from rasterio.transform import rowcol
import geopandas as gpd
from scipy.ndimage import distance_transform_edt, uniform_filter
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
LULC_DIR   = Path("data/processed/lulc_production")
CA_DIR     = Path("data/processed/ca_ann")
PROC_DIR   = Path("data/processed")
AHP_DIR    = Path("data/processed/ahp")
MAPS_DIR   = Path("maps")
AHP_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference raster (use LULC 2024 as template) ──────────
REF_RASTER = LULC_DIR / "lulc_2024.tif"


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
def load_ref():
    """Load reference raster metadata."""
    with rasterio.open(REF_RASTER) as src:
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width
    return profile, transform, crs, H, W


def normalise(arr, invert=False):
    """Normalise array to 0-1. invert=True → high values become low."""
    arr = arr.astype(float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    out = (arr - mn) / (mx - mn + 1e-10)
    return np.clip(1 - out if invert else out, 0, 1)


def save_raster(data, profile, path, dtype="float32"):
    p = profile.copy()
    p.update({"count": 1, "dtype": dtype,
               "nodata": -9999, "driver": "GTiff"})
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"   💾 {path}")


# ════════════════════════════════════════════════════════════
# STEP 1 — COMPUTE 6 CRITERIA RASTERS
# ════════════════════════════════════════════════════════════

def criterion_1_population_density(H, W, profile):
    """
    Population density proxy — built-up neighbourhood density from LULC 2024.
    Higher built-up density around a cell → more people → higher demand.
    Uses a 15×15 pixel (~450m) moving window.
    """
    print("   🔢 C1: Population density (built-up density proxy)...")
    with rasterio.open(LULC_DIR / "lulc_2024.tif") as src:
        lulc = src.read(1)

    built = (lulc == 1).astype(float)
    # 15-pixel window ≈ 450m radius at 30m resolution
    density = uniform_filter(built, size=15)
    result  = normalise(density)
    save_raster(result, profile, AHP_DIR / "c1_population_density.tif")
    return result


def criterion_2_distance_hospitals(H, W, profile, transform, crs):
    """
    Distance to existing hospitals — farther = better (service gap).
    Inverted: high score = far from existing hospitals = underserved area.
    """
    print("   🏥 C2: Distance to existing hospitals...")
    hospitals = gpd.read_file(PROC_DIR / "hospitals_osm.gpkg")
    hospitals = hospitals[
        hospitals.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon"])
    ].copy()
    hospitals = hospitals.to_crs(crs)

    # Rasterise hospital locations
    shapes = [(geom, 1) for geom in hospitals.geometry if geom is not None]
    hosp_raster = rasterize(
        shapes, out_shape=(H, W),
        transform=transform, fill=0, dtype=np.uint8
    )

    # Distance transform
    dist = distance_transform_edt(1 - hosp_raster)
    # Invert: far from hospitals = high score (underserved)
    result = normalise(dist)
    save_raster(result, profile, AHP_DIR / "c2_hospital_distance.tif")
    return result


def criterion_3_growth_hotspot(profile):
    """
    Proximity to predicted growth hotspots (2030-2035).
    High = cell is in or near a growth hotspot → future demand.
    """
    print("   📈 C3: Growth hotspot proximity...")
    with rasterio.open(CA_DIR / "growth_hotspots_2030_2035.tif") as src:
        hotspot = src.read(1).astype(float)

    # Smooth hotspot with Gaussian-like window for proximity effect
    hotspot_smooth = uniform_filter(hotspot, size=11)
    result = normalise(hotspot_smooth)
    save_raster(result, profile, AHP_DIR / "c3_growth_hotspot.tif")
    return result


def criterion_4_road_accessibility(H, W, profile, transform, crs):
    """
    Road accessibility — distance to road network.
    Inverted: closer to roads = higher accessibility = better for hospital.
    """
    print("   🛣️  C4: Road accessibility...")
    roads_path = PROC_DIR / "roads" / "vellore_roads.gpkg"

    try:
        # Try loading edge layer
        edges = gpd.read_file(roads_path, layer="edges")
        edges = edges.to_crs(crs)
        shapes = [(geom, 1) for geom in edges.geometry if geom is not None]
    except Exception:
        # Fallback: load any layer
        try:
            roads = gpd.read_file(roads_path)
            roads = roads.to_crs(crs)
            shapes = [(geom, 1) for geom in roads.geometry if geom is not None]
        except Exception as e:
            print(f"   ⚠️  Road file error: {e} — using built-up proxy")
            # Proxy: built-up areas are near roads in Indian cities
            with rasterio.open(LULC_DIR / "lulc_2024.tif") as src:
                lulc = src.read(1)
            built = (lulc == 1).astype(np.uint8)
            dist  = distance_transform_edt(1 - built)
            return normalise(dist, invert=True)

    road_raster = rasterize(
        shapes, out_shape=(H, W),
        transform=transform, fill=0, dtype=np.uint8
    )
    dist   = distance_transform_edt(1 - road_raster)
    result = normalise(dist, invert=True)  # closer = better
    save_raster(result, profile, AHP_DIR / "c4_road_accessibility.tif")
    return result


def criterion_5_environmental_safety(H, W, profile):
    """
    Environmental safety — avoid water bodies and steep areas.
    Water pixels = 0 (unsafe), non-water = 1 (safe).
    Also penalise areas very close to Palar river (flood risk).
    """
    print("   🌊 C5: Environmental safety (flood/water avoidance)...")
    with rasterio.open(LULC_DIR / "lulc_2024.tif") as src:
        lulc = src.read(1)

    # Water pixels get score 0
    water_mask = (lulc == 3).astype(float)
    # Buffer around water (5 pixels ≈ 150m) gets reduced score
    water_buffered = uniform_filter(water_mask, size=11)
    safety = 1 - np.clip(water_buffered * 3, 0, 1)
    result = normalise(safety)
    save_raster(result, profile, AHP_DIR / "c5_environmental_safety.tif")
    return result


def criterion_6_land_suitability(H, W, profile):
    """
    Land suitability — non-built, non-water areas in 2024 LULC.
    Available land: vegetation or bare land → suitable for new hospital.
    Built-up = already developed, Water = unusable.
    """
    print("   🌱 C6: Land suitability (available land)...")
    with rasterio.open(LULC_DIR / "lulc_2024.tif") as src:
        lulc = src.read(1)

    # Suitable: vegetation (2) or bare land (4)
    suitable = np.isin(lulc, [2, 4]).astype(float)
    # Slight preference for bare land (cheaper to develop)
    bare_bonus = (lulc == 4).astype(float) * 0.2
    result = normalise(suitable + bare_bonus)
    save_raster(result, profile, AHP_DIR / "c6_land_suitability.tif")
    return result


# ════════════════════════════════════════════════════════════
# STEP 2 — AHP WEIGHTS
# ════════════════════════════════════════════════════════════

def compute_ahp_weights():
    """
    AHP Pairwise Comparison Matrix (6×6).
    Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme preference.

    Criteria:
      C1 Population density       — most critical (future patients)
      C2 Distance to hospitals    — second (service gap identification)
      C3 Growth hotspot proximity — third (future demand)
      C4 Road accessibility       — fourth (reachability)
      C5 Environmental safety     — fifth (site constraints)
      C6 Land suitability         — sixth (development feasibility)
    """

    # Pairwise matrix A[i,j] = how much more important i is than j
    A = np.array([
        #C1   C2   C3   C4   C5   C6
        [1,   2,   2,   3,   4,   5],   # C1 Pop density
        [1/2, 1,   1,   2,   3,   4],   # C2 Hosp distance
        [1/2, 1,   1,   2,   3,   4],   # C3 Growth hotspot
        [1/3, 1/2, 1/2, 1,   2,   3],   # C4 Road access
        [1/4, 1/3, 1/3, 1/2, 1,   2],   # C5 Env safety
        [1/5, 1/4, 1/4, 1/3, 1/2, 1],  # C6 Land suitability
    ], dtype=float)

    n = A.shape[0]

    # Step 1: Normalise each column
    col_sums = A.sum(axis=0)
    A_norm   = A / col_sums

    # Step 2: Compute weights (row means of normalised matrix)
    weights  = A_norm.mean(axis=1)

    # Step 3: Compute consistency
    Aw         = A @ weights
    lambda_max = (Aw / weights).mean()
    CI         = (lambda_max - n) / (n - 1)

    # Random Index values (Saaty 1987)
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
                5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    RI = RI_table[n]
    CR = CI / RI

    return weights, CR, lambda_max, CI, A


def print_ahp_report(weights, CR, lambda_max, CI):
    names = [
        "C1 Population Density  ",
        "C2 Hospital Distance   ",
        "C3 Growth Hotspot      ",
        "C4 Road Accessibility  ",
        "C5 Environmental Safety",
        "C6 Land Suitability    ",
    ]
    print("\n   📊 AHP Weight Report:")
    print("   " + "─" * 42)
    for name, w in zip(names, weights):
        bar = "█" * int(w * 60)
        print(f"   {name}: {w:.4f}  {bar}")
    print("   " + "─" * 42)
    print(f"   λ_max:  {lambda_max:.4f}")
    print(f"   CI:     {CI:.4f}")
    print(f"   CR:     {CR:.4f}  "
          f"{'✅ PASS (< 0.10)' if CR < 0.10 else '⚠️  FAIL — recalibrate matrix'}")
    print(f"   Total weight: {weights.sum():.4f}")


# ════════════════════════════════════════════════════════════
# STEP 3 — WEIGHTED OVERLAY
# ════════════════════════════════════════════════════════════

def weighted_overlay(criteria_list, weights):
    """Combine all criteria into single suitability score (0-1)."""
    suitability = np.zeros_like(criteria_list[0], dtype=float)
    for criterion, weight in zip(criteria_list, weights):
        suitability += criterion * weight
    return np.clip(suitability, 0, 1)


def classify_suitability(suitability):
    """
    Classify suitability into 3 zones:
      High   (3): score ≥ 0.65
      Medium (2): score 0.40 – 0.65
      Low    (1): score < 0.40
    """
    classified = np.ones_like(suitability, dtype=np.uint8)  # Low
    classified[suitability >= 0.40] = 2  # Medium
    classified[suitability >= 0.65] = 3  # High
    return classified


# ════════════════════════════════════════════════════════════
# STEP 4 — MAPS
# ════════════════════════════════════════════════════════════

def plot_criteria_grid(criteria_list, names):
    """6-panel grid showing all criteria layers."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    cmap = plt.cm.RdYlGn  # red=low, green=high suitability

    for ax, crit, name in zip(axes, criteria_list, names):
        im = ax.imshow(crit, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("AHP Criteria Layers — Vellore Hospital Suitability",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MAPS_DIR / "ahp_criteria_grid.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("   🗺️  maps/ahp_criteria_grid.png")


def plot_suitability(suitability, classified):
    """Two-panel: continuous score + classified High/Medium/Low."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Panel 1: continuous score
    cmap_cont = LinearSegmentedColormap.from_list(
        "suit", ["#D73027", "#FEE08B", "#1A9850"])
    im = ax1.imshow(suitability, cmap=cmap_cont, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04,
                 label="Suitability Score (0–1)")
    ax1.set_title("Hospital Site Suitability Score\nVellore Urban Core",
                  fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Panel 2: classified
    class_cmap = mcolors.ListedColormap(["#D73027", "#FEE08B", "#1A9850"])
    class_norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], class_cmap.N)
    ax2.imshow(classified, cmap=class_cmap, norm=class_norm)

    counts = {
        "High (≥0.65)":   int(np.sum(classified == 3)),
        "Medium (0.40–0.65)": int(np.sum(classified == 2)),
        "Low (<0.40)":    int(np.sum(classified == 1)),
    }
    total = sum(counts.values())
    stats = "\n".join([f"{k}: {v:,}  ({v/total*100:.1f}%)"
                       for k, v in counts.items()])
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax2.set_title("Suitability Classification\nHigh / Medium / Low",
                  fontsize=13, fontweight="bold")
    ax2.axis("off")

    patches = [
        mpatches.Patch(color="#1A9850", label="High Suitability (≥ 0.65)"),
        mpatches.Patch(color="#FEE08B", label="Medium (0.40–0.65)"),
        mpatches.Patch(color="#D73027", label="Low (< 0.40)"),
    ]
    ax2.legend(handles=patches, loc="lower right",
               fontsize=11, framealpha=0.9)

    plt.suptitle("Hospital Site Suitability Analysis — Vellore 2024→2035",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MAPS_DIR / "suitability_map.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("   🗺️  maps/suitability_map.png")

    return counts


def plot_weights_bar(weights, CR):
    """Bar chart of AHP weights for IEEE paper."""
    names = ["Population\nDensity", "Hospital\nDistance",
             "Growth\nHotspot", "Road\nAccess",
             "Env.\nSafety", "Land\nSuitability"]
    colors = ["#E74C3C", "#3498DB", "#E67E22",
              "#2ECC71", "#9B59B6", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, weights * 100, color=colors, edgecolor="black",
                  linewidth=0.8, alpha=0.9)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{w*100:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Weight (%)", fontsize=12)
    ax.set_title(f"AHP Criteria Weights — Vellore Hospital Suitability\n"
                 f"Consistency Ratio (CR) = {CR:.4f} < 0.10 ✅",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(weights) * 120)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(MAPS_DIR / "ahp_weights_chart.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("   🗺️  maps/ahp_weights_chart.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🏥 AHP Suitability Analysis — Vellore")
    print("=" * 60)

    # Load reference metadata
    profile, transform, crs, H, W = load_ref()
    print(f"✅ Reference raster: {H}×{W} px  CRS={crs}")

    # ── STEP 1: Compute criteria ──────────────────────────
    print("\n📐 Computing criteria rasters...")
    c1 = criterion_1_population_density(H, W, profile)
    c2 = criterion_2_distance_hospitals(H, W, profile, transform, crs)
    c3 = criterion_3_growth_hotspot(profile)
    c4 = criterion_4_road_accessibility(H, W, profile, transform, crs)
    c5 = criterion_5_environmental_safety(H, W, profile)
    c6 = criterion_6_land_suitability(H, W, profile)

    criteria_list = [c1, c2, c3, c4, c5, c6]
    criteria_names = [
        "C1: Population Density",
        "C2: Hospital Distance (inverted)",
        "C3: Growth Hotspot Proximity",
        "C4: Road Accessibility",
        "C5: Environmental Safety",
        "C6: Land Suitability",
    ]

    print("\n✅ All 6 criteria computed")
    for name, crit in zip(criteria_names, criteria_list):
        print(f"   {name}  "
              f"min={crit.min():.3f}  max={crit.max():.3f}  "
              f"mean={crit.mean():.3f}")

    # ── STEP 2: AHP weights ───────────────────────────────
    print("\n⚖️  Running AHP...")
    weights, CR, lambda_max, CI, A = compute_ahp_weights()
    print_ahp_report(weights, CR, lambda_max, CI)

    if CR >= 0.10:
        print("\n   ⚠️  CR ≥ 0.10 — matrix is inconsistent!")
        print("   Adjust pairwise values and re-run.")
        return

    # ── STEP 3: Weighted overlay ──────────────────────────
    print("\n🔢 Computing weighted overlay...")
    suitability = weighted_overlay(criteria_list, weights)
    print(f"   Suitability range: [{suitability.min():.4f}, "
          f"{suitability.max():.4f}]  mean={suitability.mean():.4f}")

    # ── STEP 4: Classify ──────────────────────────────────
    classified = classify_suitability(suitability)

    # ── STEP 5: Save rasters ──────────────────────────────
    print("\n💾 Saving rasters...")
    save_raster(suitability, profile,
                AHP_DIR / "suitability_score.tif")
    p = profile.copy()
    p.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(AHP_DIR / "suitability_classified.tif", "w", **p) as dst:
        dst.write(classified, 1)
    print(f"   💾 {AHP_DIR}/suitability_classified.tif")

    # ── STEP 6: Maps ──────────────────────────────────────
    print("\n🗺️  Generating maps...")
    plot_criteria_grid(criteria_list, criteria_names)
    counts = plot_suitability(suitability, classified)
    plot_weights_bar(weights, CR)

    # ── Summary ───────────────────────────────────────────
    total = sum(counts.values())
    area_per_px = 30 * 30 / 1e6  # km² per pixel (30m resolution)

    print(f"\n{'='*60}")
    print("📊 Suitability Summary:")
    print(f"   {'Zone':<25} {'Pixels':>8}  {'%':>6}  {'Area (km²)':>10}")
    print("   " + "─" * 54)
    for zone, n in counts.items():
        area = n * area_per_px
        print(f"   {zone:<25} {n:>8,}  "
              f"{n/total*100:>5.1f}%  {area:>9.2f}")

    high_px = counts["High (≥0.65)"]
    print(f"\n   🟢 High suitability area: "
          f"{high_px * area_per_px:.2f} km²")
    print(f"   → These are your candidate hospital zones")

    print(f"\n   AHP Weights used:")
    names_short = ["Pop.Density", "Hosp.Dist", "Growth",
                   "Roads", "EnvSafety", "LandSuit"]
    for n, w in zip(names_short, weights):
        print(f"   {n:<12}: {w:.4f}  ({w*100:.1f}%)")

    print(f"\n{'='*60}")
    print("✅ Stage 3 COMPLETE — AHP Suitability Analysis")
    print()
    print("📁 Outputs:")
    print("   data/processed/ahp/suitability_score.tif")
    print("   data/processed/ahp/suitability_classified.tif")
    print("   data/processed/ahp/c1_*.tif  through  c6_*.tif")
    print("   maps/suitability_map.png")
    print("   maps/ahp_criteria_grid.png")
    print("   maps/ahp_weights_chart.png")
    print()
    print("➡️  Next: run python src/05_site_recommendation.py")


if __name__ == "__main__":
    main()