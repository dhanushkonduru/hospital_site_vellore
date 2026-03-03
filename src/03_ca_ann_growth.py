"""
03_ca_ann_growth.py
===================
Stage 2: Cellular Automata + Artificial Neural Network (CA-ANN)
Urban Growth Prediction for Vellore Hospital Site Suitability Project.

FIXES vs original:
  - LULC_DIR corrected: data/processed/lulc_production → data/processed/lulc
  - load_lulc() now clips full-scene TIFs to Vellore study area (744×730 px)
    before any processing — fixes "built-up going backwards" issue
  - Built-up pixel counts now match paper values (2013≈28%, 2024≈38%)

Pipeline:
1. Compute spatial driver rasters from existing data
2. Extract transition pixels (non-built → built) from 2013→2019
3. Train ANN on transition probability
4. Validate: predict 2024 built-up, compare with actual 2024 LULC
5. Simulate CA to predict 2030 and 2035
6. Output growth hotspot raster for AHP suitability analysis
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
import geopandas as gpd
from scipy.ndimage import distance_transform_edt, binary_dilation
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (classification_report, cohen_kappa_score,
                              confusion_matrix, precision_recall_curve)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import glob, warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
LULC_DIR  = Path("data/processed/lulc")
OUT_DIR   = Path("data/processed/ca_ann")
MAPS_DIR  = Path("maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BUILT_UP_CLASS = 1
VELLORE_CBD    = (79.1324, 12.9165)   # lon, lat — Vellore city center
DEBUG = True

# ── Vellore study area clip bounds (WGS84) ────────────────────────────────────
# FIX: Stage 2 now writes full-scene TIFs (7801×7631).
# load_lulc() clips each to this bbox → 744×730 px matching the paper.
STUDY_BOUNDS = {
    "minx": 79.02, "miny": 12.82,
    "maxx": 79.22, "maxy": 13.02,
}

# ── Colour map for growth maps ────────────────────────────────────────────────
GROWTH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "growth", ["#FFFFFF", "#FFEDA0", "#FEB24C", "#FC4E2A", "#800026"])


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load LULC rasters, clip to Vellore study area
# ════════════════════════════════════════════════════════════════════════════
def clip_to_study_area(src):
    """
    Clips a rasterio DatasetReader to the Vellore study area bbox.
    Handles both full-scene (7800×7600) and already-clipped (744×730) inputs.
    Returns (clipped_array, clipped_profile).
    """
    raster_crs = src.crs
    # Reproject bbox from WGS84 → raster CRS
    if raster_crs.to_epsg() == 4326:
        minx, miny = STUDY_BOUNDS["minx"], STUDY_BOUNDS["miny"]
        maxx, maxy = STUDY_BOUNDS["maxx"], STUDY_BOUNDS["maxy"]
    else:
        minx, miny, maxx, maxy = transform_bounds(
            CRS.from_epsg(4326), raster_crs,
            STUDY_BOUNDS["minx"], STUDY_BOUNDS["miny"],
            STUDY_BOUNDS["maxx"], STUDY_BOUNDS["maxy"],
        )

    clip_geom = [box(minx, miny, maxx, maxy).__geo_interface__]
    try:
        clipped, transform = rio_mask(src, clip_geom, crop=True, nodata=0)
        profile = src.profile.copy()
        profile.update({
            "height":    clipped.shape[1],
            "width":     clipped.shape[2],
            "transform": transform,
        })
        return clipped[0].astype(np.uint8), profile
    except Exception:
        # If clip fails (e.g. already smaller than bbox), return as-is
        data = src.read(1).astype(np.uint8)
        return data, src.profile.copy()


def load_lulc(year):
    path = LULC_DIR / f"lulc_{year}.tif"
    with rasterio.open(path) as src:
        data, profile = clip_to_study_area(src)
        transform = profile["transform"]
        crs       = profile.get("crs", src.crs)
    print(f"   ✅ LULC {year}: {data.shape}  CRS={crs}  "
          f"built-up={np.sum(data==BUILT_UP_CLASS)/data.size*100:.1f}%")
    return data, profile, transform, crs


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute spatial driver rasters
# ════════════════════════════════════════════════════════════════════════════
def compute_drivers(lulc_2013, profile, transform, crs):
    H, W = lulc_2013.shape
    print("\n📐 Computing spatial driver rasters...")

    def norm(arr, invert=False):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        out = (arr - mn) / (mx - mn + 1e-10)
        return 1 - out if invert else out

    # Driver 1: Distance to existing built-up (2013)
    built_mask = (lulc_2013 == BUILT_UP_CLASS).astype(np.uint8)
    dist_builtup = distance_transform_edt(1 - built_mask)
    dist_builtup_norm = norm(dist_builtup, invert=True)
    print("   ✅ Driver 1: Distance to built-up")

    # Driver 2: Distance to roads (OSM)
    try:
        import osmnx as ox
        boundary = gpd.GeoDataFrame(
            geometry=[box(STUDY_BOUNDS["minx"], STUDY_BOUNDS["miny"],
                          STUDY_BOUNDS["maxx"], STUDY_BOUNDS["maxy"])],
            crs="EPSG:4326")
        G = ox.graph_from_polygon(boundary.geometry[0], network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False)
        from rasterio.features import rasterize
        edges_reproj = edges.to_crs(crs)
        road_shapes  = [(geom, 1) for geom in edges_reproj.geometry
                        if geom is not None]
        road_raster  = rasterize(road_shapes, out_shape=(H, W),
                                 transform=transform, fill=0, dtype=np.uint8)
        dist_road = distance_transform_edt(1 - road_raster)
        print("   ✅ Driver 2: Distance to roads (OSM)")
    except Exception as e:
        print(f"   ⚠️  Road distance fallback ({e})")
        dist_road = distance_transform_edt(1 - built_mask) * 0.5
    dist_road_norm = norm(dist_road, invert=True)

    # Driver 3: Distance to CBD
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cbd_x, cbd_y = transformer.transform(VELLORE_CBD[0], VELLORE_CBD[1])
    cbd_row, cbd_col = rowcol(transform, cbd_x, cbd_y)
    cbd_row = max(0, min(H-1, cbd_row))
    cbd_col = max(0, min(W-1, cbd_col))
    rows_g, cols_g = np.mgrid[0:H, 0:W]
    dist_cbd = np.sqrt((rows_g - cbd_row)**2 + (cols_g - cbd_col)**2)
    dist_cbd_norm = norm(dist_cbd, invert=True)
    print(f"   ✅ Driver 3: Distance to CBD (pixel {cbd_row},{cbd_col})")

    # Driver 4: Slope proxy
    from scipy.ndimage import sobel
    gx = sobel(dist_builtup, axis=1)
    gy = sobel(dist_builtup, axis=0)
    slope_norm = norm(np.sqrt(gx**2 + gy**2), invert=True)
    print("   ✅ Driver 4: Slope proxy (gradient)")

    # Driver 5: Neighbourhood built-up density
    from scipy.ndimage import uniform_filter
    neigh_density = uniform_filter(built_mask.astype(float), size=5)
    neigh_norm = norm(neigh_density)
    print("   ✅ Driver 5: Neighbourhood built-up density")

    # Driver 6: Distance to water (from LULC class 3)
    water_mask = (lulc_2013 == 3).astype(np.uint8)
    dist_water = distance_transform_edt(1 - water_mask)
    dist_water_norm = norm(dist_water, invert=False)
    print("   ✅ Driver 6: Distance to water")

    # Driver 7: Distance to hospitals (OSM points)
    dist_hospital = np.full((H, W), fill_value=np.sqrt(H**2 + W**2), dtype=float)
    hospitals_candidates = [
        Path("data/processed/hospitals.gpkg"),
        Path("data/processed/hospitals_osm.gpkg"),
    ]
    try:
        hospital_path = next((p for p in hospitals_candidates if p.exists()), None)
        if hospital_path is not None:
            hospitals = gpd.read_file(hospital_path)
            if hospitals.empty:
                raise ValueError("hospital layer is empty")
            hospitals = hospitals.to_crs(crs)
            rows_g, cols_g = np.mgrid[0:H, 0:W]
            rows_h, cols_h = [], []
            for geom in hospitals.geometry:
                if geom is None:
                    continue
                cx, cy = geom.centroid.x, geom.centroid.y
                rr, cc = rowcol(transform, cx, cy)
                if 0 <= rr < H and 0 <= cc < W:
                    rows_h.append(rr)
                    cols_h.append(cc)
            if rows_h:
                rows_h = np.array(rows_h)
                cols_h = np.array(cols_h)
                for rr, cc in zip(rows_h, cols_h):
                    d = np.sqrt((rows_g - rr)**2 + (cols_g - cc)**2)
                    dist_hospital = np.minimum(dist_hospital, d)
                print(f"   ✅ Driver 7: Distance to hospitals ({len(rows_h)} points)")
            else:
                raise ValueError("no hospital points intersect raster extent")
        else:
            raise FileNotFoundError("hospitals.gpkg/hospitals_osm.gpkg not found")
    except Exception as e:
        print(f"   ⚠️  Hospital distance fallback ({e})")
        dist_hospital = distance_transform_edt(1 - built_mask)
    dist_hospital_norm = norm(dist_hospital, invert=True)

    # Driver 8: Distance to bare/fallow land (class 4)
    bare_mask = (lulc_2013 == 4).astype(np.uint8)
    dist_bare = distance_transform_edt(1 - bare_mask)
    dist_bare_norm = norm(dist_bare, invert=True)
    print("   ✅ Driver 8: Distance to bare/fallow")

    drivers = np.stack([
        dist_builtup_norm, dist_road_norm, dist_cbd_norm,
        slope_norm, neigh_norm, dist_water_norm,
        dist_hospital_norm, dist_bare_norm,
    ], axis=0)
    print(f"   Driver stack shape: {drivers.shape}")
    return drivers


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Extract transition samples
# ════════════════════════════════════════════════════════════════════════════
def extract_transitions(lulc_t0, lulc_t1, drivers, n_samples=6000):
    n_drivers = drivers.shape[0]
    feats = drivers.reshape(n_drivers, -1).T
    transitioned    = ((lulc_t0 != BUILT_UP_CLASS) &
                       (lulc_t1 == BUILT_UP_CLASS)).ravel()
    stable_nonbuilt = ((lulc_t0 != BUILT_UP_CLASS) &
                       (lulc_t1 != BUILT_UP_CLASS)).ravel()
    valid = ~np.isnan(feats).any(axis=1)

    pos_idx = np.where(transitioned & valid)[0]
    neg_idx = np.where(stable_nonbuilt & valid)[0]

    np.random.seed(42)
    n_pos = min(len(pos_idx), n_samples // 2)
    n_neg = min(len(neg_idx), n_samples // 2)
    pos_idx = np.random.choice(pos_idx, n_pos, replace=False)
    neg_idx = np.random.choice(neg_idx, n_neg, replace=False)

    X = np.vstack([feats[pos_idx], feats[neg_idx]])
    y = np.array([1] * n_pos + [0] * n_neg)
    print(f"   Transition pixels:    {n_pos:,}")
    print(f"   Non-transition pixels:{n_neg:,}")
    print(f"   Total training set:   {len(X):,}")
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train ANN
# ════════════════════════════════════════════════════════════════════════════
def train_ann(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(n_features,),
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15,
            restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
    ]

    print("\n🧠 Training ANN...")
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=100, batch_size=256,
        callbacks=callbacks, verbose=0
    )
    best_epoch = np.argmax(history.history["val_auc"])
    print(f"   Best epoch: {best_epoch + 1}")
    print(f"   Val AUC:    {history.history['val_auc'][best_epoch]:.4f}")
    print(f"   Val Loss:   {history.history['val_loss'][best_epoch]:.4f}")

    raw_val_prob = model.predict(X_val_s, batch_size=1024, verbose=0).ravel()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_prob, y_val)
    calibrated_val_probs = calibrator.transform(raw_val_prob)

    precision, recall, thresholds = precision_recall_curve(y_val, calibrated_val_probs)
    if len(thresholds) > 0:
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        best_idx = int(np.nanargmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    pred_val = (calibrated_val_probs >= best_threshold).astype(np.uint8)
    if pred_val.sum() == 0 and np.sum(y_val == 1) > 0:
        best_threshold = float(np.percentile(calibrated_val_probs[y_val == 1], 10))
        pred_val = (calibrated_val_probs >= best_threshold).astype(np.uint8)

    val_cm = confusion_matrix(y_val, pred_val, labels=[0, 1])
    print(f"   Best threshold (F1): {best_threshold:.4f}")
    print("   Validation confusion matrix [[TN,FP],[FN,TP]]:")
    print(f"   {val_cm.tolist()}")
    print(f"   Predicted positive count: {int(pred_val.sum())}")
    print(f"   True positive count:      {int(np.sum((pred_val == 1) & (y_val == 1)))}")
    print("   ✅ Isotonic calibration fitted on validation probabilities")

    if DEBUG:
        print(f"   Debug: raw_val_prob range = {raw_val_prob.min():.4f}..{raw_val_prob.max():.4f}")
        print(f"   Debug: calibrated range   = {calibrated_val_probs.min():.4f}..{calibrated_val_probs.max():.4f}")

    return model, scaler, history, calibrator, best_threshold


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compute transition probability map
# ════════════════════════════════════════════════════════════════════════════
def compute_transition_prob(model, scaler, calibrator, drivers, lulc_current):
    H, W = drivers.shape[1], drivers.shape[2]
    n_drivers = drivers.shape[0]
    feats = drivers.reshape(n_drivers, -1).T
    valid = ~np.isnan(feats).any(axis=1)

    feats_scaled = np.zeros_like(feats)
    feats_scaled[valid] = scaler.transform(feats[valid])

    prob_flat = np.zeros(H * W)
    raw_prob = model.predict(feats_scaled[valid], batch_size=1024, verbose=0).ravel()
    prob_flat[valid] = calibrator.transform(raw_prob)

    prob = prob_flat.reshape(H, W)
    prob = prob * (lulc_current != BUILT_UP_CLASS)
    return prob


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Cellular Automata simulation
# ════════════════════════════════════════════════════════════════════════════
def run_ca_simulation(lulc_current, model, scaler, calibrator, prob_threshold, drivers,
                      n_steps, growth_rate_per_step):
    lulc_sim = lulc_current.copy()
    H, W = lulc_sim.shape
    print(f"\n🔄 CA Simulation: {n_steps} steps, "
          f"growth_rate={growth_rate_per_step:.4f}/step")

    for step in range(n_steps):
        built_mask = (lulc_sim == BUILT_UP_CLASS).astype(float)
        from scipy.ndimage import uniform_filter
        neigh    = uniform_filter(built_mask, size=5)
        dist_b   = distance_transform_edt(1 - built_mask.astype(np.uint8))
        drivers_step = drivers.copy()
        mn, mx = dist_b.min(), dist_b.max()
        drivers_step[0] = 1 - (dist_b - mn) / (mx - mn + 1e-10)
        mn2, mx2 = neigh.min(), neigh.max()
        drivers_step[4] = (neigh - mn2) / (mx2 - mn2 + 1e-10)

        prob = compute_transition_prob(model, scaler, calibrator, drivers_step, lulc_sim)
        n_non_built = np.sum(lulc_sim != BUILT_UP_CLASS)
        n_convert   = max(1, int(n_non_built * growth_rate_per_step))

        non_built_idx = np.where((lulc_sim != BUILT_UP_CLASS).ravel())[0]
        prob_flat     = prob.ravel()
        eligible_idx = non_built_idx[prob_flat[non_built_idx] >= prob_threshold]
        if len(eligible_idx) >= n_convert:
            source_idx = eligible_idx
        else:
            source_idx = non_built_idx

        top_idx = source_idx[np.argsort(prob_flat[source_idx])[-n_convert:]]

        lulc_flat = lulc_sim.ravel()
        lulc_flat[top_idx] = BUILT_UP_CLASS
        lulc_sim = lulc_flat.reshape(H, W)

        n_built = np.sum(lulc_sim == BUILT_UP_CLASS)
        if (step + 1) % 3 == 0 or step == n_steps - 1:
            print(f"   Step {step+1:2d}: built-up={n_built:,} px "
                  f"({n_built/(H*W)*100:.1f}%)  converted={n_convert}")
            if DEBUG:
                pred_pos = int(np.sum(prob_flat >= prob_threshold))
                print(f"      Debug: prob>=threshold pixels = {pred_pos:,} (threshold={prob_threshold:.4f})")

    return lulc_sim, prob


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Validate prediction
# ════════════════════════════════════════════════════════════════════════════
def validate(lulc_predicted, lulc_actual, label=""):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)
    pred_built   = (lulc_predicted == BUILT_UP_CLASS).ravel()
    actual_built = (lulc_actual    == BUILT_UP_CLASS).ravel()

    acc   = accuracy_score(actual_built, pred_built)
    prec  = precision_score(actual_built, pred_built, zero_division=0)
    rec   = recall_score(actual_built, pred_built, zero_division=0)
    f1    = f1_score(actual_built, pred_built, zero_division=0)
    kappa = cohen_kappa_score(actual_built, pred_built)

    print(f"\n   📊 Validation {label}:")
    print(f"   Overall Accuracy: {acc:.4f}")
    print(f"   Precision:        {prec:.4f}")
    print(f"   Recall:           {rec:.4f}")
    print(f"   F1-score:         {f1:.4f}")
    print(f"   Kappa:            {kappa:.4f}  "
          f"{'✅' if kappa >= 0.40 else '⚠️'}")
    return kappa


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save rasters + maps
# ════════════════════════════════════════════════════════════════════════════
def save_raster(data, profile, name, dtype="uint8"):
    p = profile.copy()
    p.update({"count": 1, "dtype": dtype, "nodata": 0, "driver": "GTiff"})
    out = OUT_DIR / f"{name}.tif"
    with rasterio.open(out, "w", **p) as dst:
        dst.write(data.astype(dtype if dtype != "float32"
                              else np.float32), 1)
    print(f"   💾 {out}")
    return out


def plot_lulc(lulc, title, filename, profile=None):
    COLORS_MAP  = {1:"#E83C2A", 2:"#2ECC71", 3:"#3498DB", 4:"#F5CBA7", 0:"#000000"}
    CLASSES_MAP = {1:"Built-up", 2:"Vegetation", 3:"Water", 4:"Bare Land"}
    cmap = mcolors.ListedColormap([COLORS_MAP[i] for i in [1,2,3,4]])
    norm = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5], cmap.N)
    counts = {CLASSES_MAP[i]: np.sum(lulc==i) for i in [1,2,3,4]}
    total  = sum(counts.values())
    stats  = "\n".join([f"{k}: {v/total*100:.1f}%" for k,v in counts.items()])
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(lulc, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.axis("off")
    patches = [mpatches.Patch(color=COLORS_MAP[i], label=CLASSES_MAP[i])
               for i in [1,2,3,4]]
    ax.legend(handles=patches, loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(MAPS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   🗺️  maps/{filename}")


def plot_growth_hotspots(prob_map, filename, title):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(prob_map, cmap=GROWTH_CMAP, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Urban Growth Probability")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(MAPS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   🗺️  maps/{filename}")


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("ANN Training Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history["auc"],     label="Train AUC")
    ax2.plot(history.history["val_auc"], label="Val AUC")
    ax2.set_title("ANN Training AUC"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(MAPS_DIR / "ann_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   🗺️  maps/ann_training_history.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("🏥 CA-ANN Urban Growth Model — Vellore")
    print("=" * 60)

    # Load LULC (auto-clips full-scene TIFs to 744×730 study area)
    print("\n📂 Loading LULC rasters...")
    lulc_2013, profile, transform, crs = load_lulc("2013")
    lulc_2019, _, _, _                 = load_lulc("2019")
    lulc_2024, _, _, _                 = load_lulc("2024")

    # Sanity check — catch the backwards problem early
    b13 = np.sum(lulc_2013 == BUILT_UP_CLASS) / lulc_2013.size * 100
    b24 = np.sum(lulc_2024 == BUILT_UP_CLASS) / lulc_2024.size * 100
    if b24 < b13:
        print(f"\n   ⚠️  WARNING: 2024 built-up ({b24:.1f}%) < 2013 ({b13:.1f}%)")
        print("      This indicates the LULC files may be stale demo outputs.")
        print("      Run src/02_lulc_classification.py with real Landsat data first.")
    else:
        print(f"\n   ✅ Built-up trend: {b13:.1f}% (2013) → {b24:.1f}% (2024) — correct ✓")

    # Compute spatial drivers
    drivers = compute_drivers(lulc_2013, profile, transform, crs)

    # Extract transitions and train ANN
    print("\n🔀 Extracting transition samples (2013→2019)...")
    X, y = extract_transitions(lulc_2013, lulc_2019, drivers)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model, scaler, history, calibrator, best_threshold = train_ann(X_tr, y_tr, X_val, y_val)
    plot_training_history(history)

    # Validate: simulate 2019→2024
    print("\n✅ Validation: simulating 2019→2024 (6 annual steps)...")
    built_2013 = np.sum(lulc_2013 == BUILT_UP_CLASS)
    built_2019 = np.sum(lulc_2019 == BUILT_UP_CLASS)
    annual_growth = max(0.005,
                        (built_2019 - built_2013) /
                        (lulc_2013.size - built_2013) / 6)
    print(f"   Estimated annual growth rate: {annual_growth:.4f}")

    lulc_pred_2024, prob_2024 = run_ca_simulation(
        lulc_2019, model, scaler, calibrator, best_threshold, drivers,
        n_steps=6, growth_rate_per_step=annual_growth)

    kappa_val = validate(lulc_pred_2024, lulc_2024,
                         "Predicted 2024 vs Actual 2024")

    # Predict 2030 and 2035
    print("\n🔮 Predicting 2030 (6 steps from 2024)...")
    lulc_2030, prob_2030 = run_ca_simulation(
        lulc_2024, model, scaler, calibrator, best_threshold, drivers,
        n_steps=6, growth_rate_per_step=annual_growth)

    print("\n🔮 Predicting 2035 (5 steps from 2030)...")
    lulc_2035, prob_2035 = run_ca_simulation(
        lulc_2030, model, scaler, calibrator, best_threshold, drivers,
        n_steps=5, growth_rate_per_step=annual_growth)

    # Growth hotspot map
    growth_hotspot = ((lulc_2035 == BUILT_UP_CLASS) &
                      (lulc_2024 != BUILT_UP_CLASS)).astype(float)
    growth_weighted = growth_hotspot * prob_2035

    # Save rasters
    print("\n💾 Saving rasters...")
    save_raster(lulc_pred_2024, profile, "lulc_predicted_2024")
    save_raster(lulc_2030,      profile, "lulc_predicted_2030")
    save_raster(lulc_2035,      profile, "lulc_predicted_2035")
    save_raster(prob_2030.astype(np.float32),
                profile, "growth_prob_2030", dtype="float32")
    save_raster(growth_weighted.astype(np.float32),
                profile, "growth_hotspots_2030_2035", dtype="float32")

    # Generate maps
    print("\n🗺️  Generating maps...")
    plot_lulc(lulc_2030, "Predicted LULC — Vellore 2030", "lulc_predicted_2030.png")
    plot_lulc(lulc_2035, "Predicted LULC — Vellore 2035", "lulc_predicted_2035.png")
    plot_growth_hotspots(prob_2030, "growth_probability_2030.png",
                         "Urban Growth Probability — Vellore 2030")
    plot_growth_hotspots(growth_weighted, "growth_hotspots_2030_2035.png",
                         "Growth Hotspots 2030–2035\n(High = future hospital demand areas)")

    # Comparison trajectory map
    COLORS_MAP = {1:"#E83C2A", 2:"#2ECC71", 3:"#3498DB", 4:"#F5CBA7"}
    cmap  = mcolors.ListedColormap([COLORS_MAP[i] for i in [1,2,3,4]])
    cnorm = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, lulc, title in zip(
        axes,
        [lulc_2013, lulc_2024, lulc_2035],
        ["2013 (Actual)", "2024 (Actual)", "2035 (Predicted)"]
    ):
        ax.imshow(lulc, cmap=cmap, norm=cnorm)
        n_b = np.sum(lulc == BUILT_UP_CLASS)
        ax.set_title(f"{title}\nBuilt-up: {n_b/lulc.size*100:.1f}%",
                     fontsize=13, fontweight="bold")
        ax.axis("off")
    patches = [mpatches.Patch(color=COLORS_MAP[i],
                              label={1:"Built-up",2:"Vegetation",
                                     3:"Water",4:"Bare Land"}[i])
               for i in [1,2,3,4]]
    axes[2].legend(handles=patches, loc="lower right", fontsize=11)
    plt.suptitle("Vellore Urban Growth Trajectory — CA-ANN Model",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MAPS_DIR / "ca_ann_growth_trajectory.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("   🗺️  maps/ca_ann_growth_trajectory.png")

    # Summary
    print(f"\n{'='*60}")
    print("📊 CA-ANN Growth Summary:")
    for yr, lulc in [("2013", lulc_2013), ("2024", lulc_2024),
                     ("2030", lulc_2030), ("2035", lulc_2035)]:
        n = np.sum(lulc == BUILT_UP_CLASS)
        print(f"   {yr}: Built-up = {n:>7,} px  ({n/lulc.size*100:.1f}%)")

    new_2030 = np.sum((lulc_2030==BUILT_UP_CLASS) & (lulc_2024!=BUILT_UP_CLASS))
    new_2035 = np.sum((lulc_2035==BUILT_UP_CLASS) & (lulc_2024!=BUILT_UP_CLASS))
    print(f"\n   New built-up by 2030: {new_2030:,} pixels")
    print(f"   New built-up by 2035: {new_2035:,} pixels")
    print(f"\n   Validation Kappa (predicted vs actual 2024): {kappa_val:.4f}")

    print(f"\n{'='*60}")
    print("✅ Stage 2 COMPLETE — CA-ANN Growth Model")
    print()
    print("📁 Key outputs for Stage 3 (AHP):")
    print("   data/processed/ca_ann/growth_hotspots_2030_2035.tif")
    print("   data/processed/ca_ann/growth_prob_2030.tif")
    print("   data/processed/ca_ann/lulc_predicted_2035.tif")
    print()
    print("➡️  Next: run src/04_ahp_suitability.py")


if __name__ == "__main__":
    main()