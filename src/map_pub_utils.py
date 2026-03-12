import warnings
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np

TARGET_CRS = "EPSG:32644"
ROOT = Path(__file__).resolve().parents[1]


def set_publication_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def raster_extent(profile, shape):
    transform = profile["transform"]
    h, w = shape
    left = transform.c
    right = transform.c + transform.a * w
    top = transform.f
    bottom = transform.f + transform.e * h
    return [left, right, bottom, top]


def _pick_label_col(gdf):
    candidates = ["NAME_3", "NAME_2", "NAME_1", "name", "NAME", "district", "taluk"]
    for col in candidates:
        if col in gdf.columns:
            return col
    return None


@lru_cache(maxsize=1)
def load_boundary_layers():
    study_path = ROOT / "data" / "processed" / "vellore_boundary.gpkg"
    study = None
    if study_path.exists():
        study = gpd.read_file(study_path).to_crs(TARGET_CRS)

    admin_candidates = [
        ROOT / "data" / "processed" / "gadm41_IND_3.gpkg",
        ROOT / "data" / "processed" / "gadm41_IND_2.gpkg",
        ROOT / "data" / "raw" / "gadm41_IND_3.gpkg",
        ROOT / "data" / "raw" / "gadm41_IND_2.gpkg",
    ]

    admin = None
    label_col = None
    for cand in admin_candidates:
        if not cand.exists():
            continue
        try:
            gdf = gpd.read_file(cand).to_crs(TARGET_CRS)
            if "NAME_2" in gdf.columns:
                sel = gdf[gdf["NAME_2"].astype(str).str.lower() == "vellore"]
                if not sel.empty:
                    admin = sel
                else:
                    admin = gdf
            else:
                admin = gdf
            label_col = _pick_label_col(admin)
            break
        except Exception:
            continue

    if admin is None:
        # Optional online fallback if osmnx is available.
        try:
            import osmnx as ox

            admin = ox.geocode_to_gdf("Vellore district, Tamil Nadu, India").to_crs(TARGET_CRS)
            label_col = _pick_label_col(admin)
        except Exception:
            admin = None
            label_col = None

    return study, admin, label_col


def add_boundary_overlays(ax, study_gdf, admin_gdf, label_col):
    if admin_gdf is not None and not admin_gdf.empty:
        admin_gdf.boundary.plot(ax=ax, edgecolor="#222222", linewidth=1.2, zorder=5)
        if label_col and label_col in admin_gdf.columns:
            for _, row in admin_gdf.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                rp = row.geometry.representative_point()
                ax.annotate(
                    str(row[label_col]),
                    xy=(rp.x, rp.y),
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    ha="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        alpha=0.6,
                        ec="none",
                    ),
                    zorder=6,
                )

    if study_gdf is not None and not study_gdf.empty:
        study_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=2.0, facecolor="none", zorder=6)


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.95, 0.93),
        xytext=(0.95, 0.87),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color="black",
        zorder=7,
    )


def add_scale_bar(ax):
    ax.add_artist(ScaleBar(dx=1, units="m", location="lower left", font_properties={"size": 8}))


def style_map_axis(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)


def add_standard_colorbar(fig, ax, im, label):
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(label, fontsize=9)
    return cbar


def save_publication_figure(fig, out_path):
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
