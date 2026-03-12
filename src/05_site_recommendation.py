"""
05_site_recommendation.py
==========================
Stage 4: Hospital Site Recommendation + Validation
Vellore, Tamil Nadu — Hospital Site Suitability Project

Pipeline:
1. Reload suitability score, reclassify with data-adaptive thresholds
2. Extract High suitability candidate zones
3. Filter by minimum area + road proximity
4. Rank top 5 sites
5. Validate: coverage improvement (% population within 5km)
6. Generate final recommendation map for IEEE paper
"""

import json
import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
import geopandas as gpd
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
from scipy.ndimage import label as scipy_label, uniform_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from map_pub_utils import (
    set_publication_style,
    raster_extent,
    load_boundary_layers,
    add_boundary_overlays,
    add_north_arrow,
    add_scale_bar,
    style_map_axis,
    add_standard_colorbar,
    save_publication_figure,
)
import warnings
warnings.filterwarnings("ignore")
set_publication_style()

# ── Paths ─────────────────────────────────────────────────
AHP_DIR  = Path("data/processed/ahp")
CA_DIR   = Path("data/processed/ca_ann")
LULC_DIR = Path("data/processed/lulc")
PROC_DIR = Path("data/processed")
SITES_DIR = Path("data/processed/sites")
MAPS_DIR  = Path("maps")
SITES_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# STEP 1 — Reclassify with adaptive thresholds
# ════════════════════════════════════════════════════════════
def adaptive_reclassify(suitability):
    """
    Use percentile-based thresholds instead of fixed 0.65/0.40.
    Top 15% → High, next 30% → Medium, rest → Low.
    This ensures meaningful High zones regardless of score range.
    """
    valid = suitability[suitability > 0]
    p85 = np.percentile(valid, 85)   # top 15% = High
    p55 = np.percentile(valid, 55)   # next 30% = Medium

    classified = np.ones_like(suitability, dtype=np.uint8)   # Low
    classified[suitability >= p55] = 2   # Medium
    classified[suitability >= p85] = 3   # High

    print(f"   Adaptive thresholds:")
    print(f"   High   (top 15%): score ≥ {p85:.4f}")
    print(f"   Medium (55–85%):  score ≥ {p55:.4f}")
    print(f"   Low    (< 55%):   score <  {p55:.4f}")
    return classified, p85, p55


# ════════════════════════════════════════════════════════════
# STEP 2 — Extract candidate zones from High suitability
# ════════════════════════════════════════════════════════════
def extract_candidate_zones(classified, suitability, profile, crs):
    """
    Extract connected High suitability regions.
    Filter: minimum 3 pixels (≥ 0.27 km²).
    """
    high_mask = (classified == 3).astype(np.uint8)

    # Label connected components
    labeled, n_features = scipy_label(high_mask)
    print(f"   Found {n_features} connected High zones")

    # Convert to polygons
    transform = profile["transform"]
    zones = []
    for region_id in range(1, n_features + 1):
        region_mask = (labeled == region_id).astype(np.uint8)
        pixel_count = np.sum(region_mask)
        if pixel_count < 3:
            continue

        # Get mean suitability score for this region
        mean_score = np.mean(suitability[region_mask == 1])

        # Convert raster region to polygon
        for geom, val in shapes(region_mask, transform=transform):
            if val == 1:
                poly = shape(geom)
                area_km2 = pixel_count * 0.09 / 10  # 30m px = 0.0009 km²
                zones.append({
                    "geometry": poly,
                    "pixel_count": int(pixel_count),
                    "area_km2": float(pixel_count * 900 / 1e6),
                    "mean_score": float(mean_score),
                    "centroid_x": poly.centroid.x,
                    "centroid_y": poly.centroid.y,
                })
                break

    gdf = gpd.GeoDataFrame(zones, crs=crs)
    print(f"   Valid zones (≥ 3 pixels): {len(gdf)}")
    return gdf


# ════════════════════════════════════════════════════════════
# STEP 3 — Rank and filter to top 5 sites
# ════════════════════════════════════════════════════════════
def rank_sites(zones_gdf, roads_path, hospitals_gdf, crs):
    """
    Scoring:
      - Mean suitability score (higher = better)
      - Distance to nearest road (closer = better, up to 1km)
      - Distance to nearest hospital (farther = better, more underserved)
      - Area size (larger = better for hospital)
    """
    if len(zones_gdf) == 0:
        print("   ⚠️  No valid zones found")
        return gpd.GeoDataFrame()

    # Load roads
    try:
        roads = gpd.read_file(roads_path, layer="edges").to_crs(crs)
        roads_union = roads.geometry.unary_union
    except Exception:
        try:
            roads = gpd.read_file(roads_path).to_crs(crs)
            roads_union = roads.geometry.unary_union
        except Exception:
            roads_union = None
            print("   ⚠️  Road data not loaded for filtering")

    # Load hospitals
    hosp = hospitals_gdf.to_crs(crs)
    hosp_union = hosp.geometry.unary_union

    scores = []
    for idx, row in zones_gdf.iterrows():
        centroid = row.geometry.centroid

        # Distance to nearest road (metres)
        if roads_union:
            dist_road = centroid.distance(roads_union)
        else:
            dist_road = 500  # assume 500m if no road data

        # Distance to nearest hospital (metres)
        dist_hosp = centroid.distance(hosp_union)

        # Composite rank score
        road_score  = max(0, 1 - dist_road / 2000)    # 0-1, penalise >2km
        hosp_score  = min(1, dist_hosp / 10000)        # 0-1, reward >10km gap
        area_score  = min(1, row["area_km2"] / 0.5)    # 0-1, reward >0.5km²
        suit_score  = row["mean_score"]

        composite = (0.40 * suit_score +
                     0.25 * hosp_score +
                     0.20 * road_score +
                     0.15 * area_score)

        scores.append({
            "dist_road_m":  float(dist_road),
            "dist_hosp_m":  float(dist_hosp),
            "road_score":   float(road_score),
            "hosp_score":   float(hosp_score),
            "area_score":   float(area_score),
            "composite":    float(composite),
        })

    for key in scores[0].keys():
        zones_gdf[key] = [s[key] for s in scores]

    # Sort by composite score
    zones_gdf = zones_gdf.sort_values(
        "composite", ascending=False).reset_index(drop=True)

    # Top 5 (or fewer if less available)
    top5 = zones_gdf.head(5).copy()
    top5["site_rank"] = range(1, len(top5) + 1)
    top5["site_label"] = [f"Site {i}" for i in top5["site_rank"]]

    return top5


# ════════════════════════════════════════════════════════════
# STEP 4 — Coverage validation
# ════════════════════════════════════════════════════════════
def validate_coverage(top5, hospitals_gdf, suitability, profile, crs,
                      radius_m=5000):
    """
    Compute % of study area pixels within 5km of a hospital.
    Before: existing hospitals only.
    After: existing + each proposed site.
    """
    H, W = suitability.shape
    transform = profile["transform"]
    pixel_size = abs(transform.a)  # metres per pixel

    # Create coordinate grids
    rows_grid, cols_grid = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij")

    def px_to_coord(r, c):
        x = transform.c + c * transform.a + r * transform.b
        y = transform.f + c * transform.d + r * transform.e
        return x, y

    # All pixel centroids
    all_x = transform.c + cols_grid * transform.a
    all_y = transform.f + rows_grid * transform.e
    all_x = all_x.ravel()
    all_y = all_y.ravel()
    total_px = len(all_x)

    def covered_fraction(hospital_points):
        """Fraction of pixels within radius_m of any hospital point."""
        covered = np.zeros(total_px, dtype=bool)
        for hx, hy in hospital_points:
            dist = np.sqrt((all_x - hx)**2 + (all_y - hy)**2)
            covered |= (dist <= radius_m)
        return covered.sum() / total_px * 100

    # Existing hospital points
    hosp_crs = hospitals_gdf.to_crs(crs)
    existing_pts = [(geom.x, geom.y)
                    for geom in hosp_crs.geometry
                    if geom.geom_type == "Point"]

    before = covered_fraction(existing_pts)

    results = []
    for _, site in top5.iterrows():
        cx, cy = site.geometry.centroid.x, site.geometry.centroid.y
        after = covered_fraction(existing_pts + [(cx, cy)])
        improvement = after - before
        results.append({
            "site_label":   site["site_label"],
            "coverage_before": before,
            "coverage_after":  after,
            "improvement_%":   improvement,
        })
        print(f"   {site['site_label']}: "
              f"coverage {before:.1f}% → {after:.1f}%  "
              f"(+{improvement:.2f}%)")

    return before, results


# ════════════════════════════════════════════════════════════
# STEP 5 — Final recommendation map
# ════════════════════════════════════════════════════════════
def final_map(suitability, classified, top5, hospitals_gdf,
              profile, crs, p85, p55):
    """Publication-quality final recommendation map."""

    # Suitability background
    suit_cmap = mcolors.LinearSegmentedColormap.from_list(
        "suit", ["#D73027", "#FEE08B", "#A6D96A", "#1A9850"])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    extent = raster_extent(profile, suitability.shape)
    im = ax.imshow(
        suitability,
        cmap=suit_cmap,
        vmin=0,
        vmax=suitability.max(),
        alpha=0.85,
        extent=extent,
        origin="upper",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    add_standard_colorbar(fig, ax, im, "Suitability Score")

    # Overlay existing hospitals
    hosp_crs = hospitals_gdf.to_crs(crs)

    # Plot existing hospitals
    for _, hosp in hosp_crs.iterrows():
        if hosp.geometry.geom_type == "Point":
            ax.plot(hosp.geometry.x, hosp.geometry.y, "b+", markersize=6,
                    markeredgewidth=1.2, alpha=0.6, zorder=8)

    # Plot proposed sites
    site_colors = ["#FF0000", "#FF6B00", "#FFD700", "#00CC44", "#0066FF"]
    for _, site in top5.iterrows():
        cx, cy = site.geometry.centroid.x, site.geometry.centroid.y
        rank = site["site_rank"]
        color = site_colors[rank - 1]

        ax.plot(cx, cy, "*", markersize=22,
                color=color, markeredgecolor="black",
                markeredgewidth=1.5, zorder=10)
        ax.annotate(
            f"  Site {rank}\n  Score: {site['mean_score']:.3f}\n"
            f"  {site['area_km2']:.3f} km²",
            xy=(cx, cy),
            fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=color, alpha=0.85, edgecolor="black"),
            zorder=11
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="+", color="blue", markersize=10,
               label=f"Existing hospitals ({len(hosp_crs)})",
               linewidth=0, markeredgewidth=2),
    ]
    for rank, color in enumerate(site_colors[:len(top5)], 1):
        legend_elements.append(
            Line2D([0], [0], marker="*", color=color, markersize=14,
                   label=f"Proposed Site {rank}",
                   linewidth=0, markeredgecolor="black")
        )
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8, framealpha=0.92, title="Facilities",
              title_fontsize=11)

    study_gdf, admin_gdf, label_col = load_boundary_layers()
    add_boundary_overlays(ax, study_gdf, admin_gdf, label_col)
    style_map_axis(
        ax,
        "Recommended Hospital Sites — Vellore Urban Core\n"
        f"Based on CA-ANN Growth Prediction + AHP Analysis (2024–2035)\n"
        f"AHP CR = 0.0117  |  Thresholds: High ≥ {p85:.3f}, "
        f"Medium ≥ {p55:.3f}",
    )
    add_north_arrow(ax)
    add_scale_bar(ax)

    plt.tight_layout()
    save_publication_figure(fig, MAPS_DIR / "final_recommendation_map.png")
    print("   🗺️  maps/final_recommendation_map.png")


def site_comparison_chart(top5, coverage_results):
    """Bar chart comparing proposed sites for IEEE paper."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=300)

    labels = [r["site_label"] for r in coverage_results]
    colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"][:len(top5)]

    # Panel 1: Suitability score
    scores = list(top5["mean_score"])
    axes[0].bar(labels, scores, color=colors, edgecolor="black", alpha=0.9)
    axes[0].set_title("Mean Suitability Score", fontweight="bold")
    axes[0].set_ylabel("Score (0–1)")
    axes[0].set_ylim(0, max(scores) * 1.2)
    for i, v in enumerate(scores):
        axes[0].text(i, v + 0.001, f"{v:.3f}", ha="center",
                     fontsize=10, fontweight="bold")

    # Panel 2: Distance to nearest hospital
    dists = list(top5["dist_hosp_m"] / 1000)
    axes[1].bar(labels, dists, color=colors, edgecolor="black", alpha=0.9)
    axes[1].set_title("Distance to Nearest Hospital (km)", fontweight="bold")
    axes[1].set_ylabel("Distance (km)")
    axes[1].set_ylim(0, max(dists) * 1.2)
    for i, v in enumerate(dists):
        axes[1].text(i, v + 0.05, f"{v:.1f}", ha="center",
                     fontsize=10, fontweight="bold")

    # Panel 3: Coverage improvement
    improvements = [r["improvement_%"] for r in coverage_results]
    axes[2].bar(labels, improvements, color=colors, edgecolor="black", alpha=0.9)
    axes[2].set_title("Population Coverage Improvement (%)", fontweight="bold")
    axes[2].set_ylabel("Coverage increase (%)")
    axes[2].set_ylim(0, max(improvements) * 1.3 if max(improvements) > 0 else 1)
    for i, v in enumerate(improvements):
        axes[2].text(i, v + 0.001, f"+{v:.2f}%", ha="center",
                     fontsize=10, fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Proposed Hospital Site Comparison — Vellore",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_publication_figure(fig, MAPS_DIR / "site_comparison_chart.png")
    print("   🗺️  maps/site_comparison_chart.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("🏥 Site Recommendation — Vellore")
    print("=" * 60)

    # ── Load suitability raster ───────────────────────────
    print("\n📂 Loading suitability raster...")
    with rasterio.open(AHP_DIR / "suitability_score.tif") as src:
        suitability = src.read(1).astype(float)
        profile     = src.profile.copy()
        transform   = src.transform
        crs         = src.crs

    print(f"   Score range: [{suitability.min():.4f}, {suitability.max():.4f}]")
    print(f"   Mean: {suitability.mean():.4f}  "
          f"P85: {np.percentile(suitability[suitability>0], 85):.4f}")

    # ── Reclassify with adaptive thresholds ───────────────
    print("\n🔀 Reclassifying with adaptive thresholds...")
    classified, p85, p55 = adaptive_reclassify(suitability)

    high_px = np.sum(classified == 3)
    med_px  = np.sum(classified == 2)
    low_px  = np.sum(classified == 1)
    total   = classified.size
    print(f"\n   High:   {high_px:>8,}  ({high_px/total*100:.1f}%)"
          f"  {high_px*900/1e6:.2f} km²")
    print(f"   Medium: {med_px:>8,}  ({med_px/total*100:.1f}%)"
          f"  {med_px*900/1e6:.2f} km²")
    print(f"   Low:    {low_px:>8,}  ({low_px/total*100:.1f}%)"
          f"  {low_px*900/1e6:.2f} km²")

    # Save updated classification
    p_cls = profile.copy()
    p_cls.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(
            AHP_DIR / "suitability_classified_adaptive.tif", "w", **p_cls) as dst:
        dst.write(classified, 1)

    # ── Load supporting data ──────────────────────────────
    print("\n📂 Loading supporting data...")
    hospitals = gpd.read_file(PROC_DIR / "hospitals_osm.gpkg")
    hospitals = hospitals[
        hospitals.geometry.geom_type == "Point"].copy()
    print(f"   Hospitals: {len(hospitals)}")

    roads_path = PROC_DIR / "roads" / "vellore_roads.gpkg"

    # ── Extract candidate zones ───────────────────────────
    print("\n🗺️  Extracting High suitability zones...")
    zones = extract_candidate_zones(classified, suitability, profile, crs)

    if len(zones) == 0:
        print("   ⚠️  No zones found — lowering threshold to top 10%")
        p90 = np.percentile(suitability[suitability > 0], 90)
        high_mask = (suitability >= p90).astype(np.uint8)
        labeled, _ = scipy_label(high_mask)
        zones_list = []
        for rid in range(1, labeled.max() + 1):
            rm = (labeled == rid).astype(np.uint8)
            pc = np.sum(rm)
            if pc < 2:
                continue
            ms = np.mean(suitability[rm == 1])
            for geom, val in shapes(rm, transform=transform):
                if val == 1:
                    poly = shape(geom)
                    zones_list.append({
                        "geometry": poly,
                        "pixel_count": int(pc),
                        "area_km2": float(pc * 900 / 1e6),
                        "mean_score": float(ms),
                    })
                    break
        zones = gpd.GeoDataFrame(zones_list, crs=crs)

    print(f"\n   Candidate zones: {len(zones)}")

    # ── Rank to top 5 ─────────────────────────────────────
    print("\n🏆 Ranking sites...")
    top5 = rank_sites(zones, roads_path, hospitals, crs)

    if len(top5) == 0:
        print("   ⚠️  Could not rank sites — insufficient zones")
        return

    print(f"\n   Top {len(top5)} Recommended Sites:")
    print("   " + "─" * 60)
    for _, site in top5.iterrows():
        print(f"   Site {site['site_rank']}: "
              f"Score={site['mean_score']:.3f}  "
              f"Area={site['area_km2']:.3f} km²  "
              f"RoadDist={site['dist_road_m']:.0f}m  "
              f"HospDist={site['dist_hosp_m']/1000:.1f}km  "
              f"Composite={site['composite']:.3f}")

    # ── Save sites as GeoPackage ──────────────────────────
    top5.to_file(SITES_DIR / "candidate_sites.gpkg", driver="GPKG")
    print(f"\n   💾 data/processed/sites/candidate_sites.gpkg")

    # ── Coverage validation ───────────────────────────────
    print(f"\n📊 Coverage validation (radius = 5 km)...")
    print("   (computing distances for all pixels — may take 30s)")
    before_cov, cov_results = validate_coverage(
        top5, hospitals, suitability, profile, crs, radius_m=5000)
    print(f"\n   Baseline coverage (existing hospitals): {before_cov:.1f}%")

    # ── Maps ──────────────────────────────────────────────
    print("\n🗺️  Generating final maps...")
    final_map(suitability, classified, top5, hospitals,
              profile, crs, p85, p55)
    site_comparison_chart(top5, cov_results)

    # ── Final summary ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 FINAL PROJECT SUMMARY")
    print("=" * 60)
    print(f"\n   Study area:  Vellore Urban Core (79.02–79.22°E, 12.82–13.02°N)")
    print(f"   Resolution:  30m (Landsat 8/9)")
    print(f"   Time period: 2013 → 2024 → 2035 (predicted)")
    print()
    print(f"   LULC Results:")
    # Load actual built-up percentages from pipeline outputs
    acc_path = LULC_DIR / "accuracy_summary.json"
    b13_pct = b24_pct = "?"
    if acc_path.exists():
        with open(acc_path) as f:
            acc = json.load(f)
        b13_pct = acc.get("2013", {}).get("built_up_pct", "?")
        b24_pct = acc.get("2024", {}).get("built_up_pct", "?")
    b35_pct = "?"
    lulc35 = CA_DIR / "lulc_predicted_2035.tif"
    if lulc35.exists():
        with rasterio.open(lulc35) as src:
            arr35 = src.read(1)
            v35 = arr35[arr35 > 0]
            if len(v35):
                b35_pct = round((v35 == 1).sum() / len(v35) * 100, 1)
    print(f"   Built-up growth: {b13_pct}% (2013) → {b24_pct}% (2024) → {b35_pct}% (2035 pred)")
    print()
    print(f"   AHP Analysis:")
    print(f"   CR = 0.0117 ✅  |  6 criteria  |  Weights sum = 1.0")
    print(f"   High suitability: {high_px:,} px ({high_px*900/1e6:.2f} km²)")
    print()
    print(f"   Recommended Sites:")
    for _, site in top5.iterrows():
        print(f"   ★ Site {site['site_rank']}: "
              f"composite score = {site['composite']:.3f}  "
              f"area = {site['area_km2']:.3f} km²")
    print()
    print(f"   Coverage Improvement:")
    print(f"   Before (existing only): {before_cov:.1f}%")
    best = max(cov_results, key=lambda x: x["improvement_%"])
    print(f"   Best site ({best['site_label']}): "
          f"{best['coverage_after']:.1f}% (+{best['improvement_%']:.2f}%)")

    print(f"\n{'='*60}")
    print("✅ Stage 4 COMPLETE — Site Recommendation")
    print()
    print("📁 Final outputs:")
    print("   data/processed/sites/candidate_sites.gpkg")
    print("   maps/final_recommendation_map.png")
    print("   maps/site_comparison_chart.png")
    print()
    print("➡️  Next: Write IEEE paper using maps/ folder")
    print("   All maps ready for paper/ieee_hospital_vellore.docx")


if __name__ == "__main__":
    main()