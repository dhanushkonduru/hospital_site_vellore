# Hospital Site Suitability Prediction — Vellore, Tamil Nadu

> Integrated CA-ANN urban growth modelling + AHP multi-criteria suitability analysis  
> to identify optimal hospital locations for Vellore's 2030–2035 growth horizon.

---

## Overview

Vellore's urban area grew from **28.2% → 38.8%** of the study region between 2013 and 2024.
Existing hospitals are concentrated in the city centre while peri-urban corridors remain
underserved. This project builds a fully automated, reproducible geospatial pipeline to
identify where new hospitals should be built before the accessibility gap widens further.

**Key result:** Site 3 (northwestern growth corridor) improves population coverage within
5 km from **77.1% → 84.6%** — a 7.44 percentage point gain from a single new hospital.

---

## Project Structure

```
hospital_site_vellore/
├── data/
│   ├── raw/
│   │   ├── vlr2013/          Landsat 8  — 31 Oct 2013  (Path 143, Row 051)
│   │   ├── vlr2019/          Landsat 8  — 17 Nov 2019
│   │   └── vlr2024/          Landsat 9  — 08 Dec 2024
│   └── processed/
│       ├── vellore_boundary.gpkg
│       ├── hospitals_osm.gpkg        123 healthcare facilities
│       ├── roads/                    59,081 OSM road edges
│       ├── lulc_production/          LULC rasters (2013 / 2019 / 2024)
│       ├── ca_ann/                   Growth predictions (2030 / 2035)
│       ├── ahp/                      Suitability criteria + score rasters
│       └── sites/                    candidate_sites.gpkg (top 5 sites)
├── maps/                             Production-quality maps (200 DPI)
├── notebooks/                        Exploratory notebooks
├── paper/                            IEEE paper draft (.docx)
├── src/
│   ├── 01_setup_study_area.py        Stage 0: Download boundaries + OSM data
│   ├── 02_lulc_classification.py     Stage 1: Random Forest LULC classifier
│   ├── 03_ca_ann_growth.py           Stage 2: CA-ANN urban growth model
│   ├── 04_ahp_suitability.py         Stage 3: AHP multi-criteria analysis
│   └── 05_site_recommendation.py     Stage 4: Site ranking + validation
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Landsat 8/9 Imagery (2013, 2019, 2024)
          │
          ▼
   Stage 1 — LULC Classification
   Random Forest · 500 trees · 12 spectral features
   4 classes: Built-up / Vegetation / Water / Bare Land
   Kappa: 0.46 – 0.76
          │
          ▼
   Stage 2 — CA-ANN Urban Growth Model
   5 spatial drivers → ANN [5→64→32→16→1] → transition probability
   CA simulation → predicted LULC 2030, 2035
   New built-up by 2035: 45,255 px (~40.7 km²)
          │
          ▼
   Stage 3 — AHP Suitability Analysis
   6 criteria layers → pairwise matrix → eigenvector weights
   CR = 0.0117 ✅  |  Weighted overlay → score raster (0–1)
          │
          ▼
   Stage 4 — Site Recommendation
   980 candidate zones → composite ranking → Top 5 sites
   Coverage validation: 77.1% → 84.6% (best site, 5 km radius)
```

---

## Quick Start

```bash
# 1. Activate environment
cd hospital_site_vellore
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline in order
python src/01_setup_study_area.py     # ~5 min  (downloads OSM data)
python src/02_lulc_classification.py  # ~10 min
python src/03_ca_ann_growth.py        # ~15 min
python src/04_ahp_suitability.py      # ~5 min
python src/05_site_recommendation.py  # ~5 min
```

All maps are saved to `maps/` and all rasters to `data/processed/`.

---

## Key Results

| Metric | Value |
|--------|-------|
| Study area | 488 km² (79.02–79.22°E, 12.82–13.02°N) |
| Resolution | 30 m (Landsat 8/9 OLI) |
| Best LULC Kappa | 0.76 (2024) |
| ANN validation AUC | 0.685 |
| AHP Consistency Ratio | 0.0117 (< 0.10 ✅) |
| High suitability area | 73.3 km² |
| Recommended sites | 5 |
| Best coverage gain | +7.44 pp (Site 3, 5 km radius) |

---

## AHP Weights

| Criterion | Weight | Source |
|-----------|--------|--------|
| Population density | 34.1% | LULC built-up density proxy |
| Distance to hospitals | 20.5% | OSM healthcare facilities |
| Growth hotspot proximity | 20.5% | CA-ANN 2030–2035 prediction |
| Road accessibility | 12.3% | OSM road network |
| Environmental safety | 7.6% | LULC water class buffer |
| Land suitability | 5.0% | LULC available land |

---

## Data Sources

| Dataset | Source | Licence |
|---------|--------|---------|
| Landsat 8/9 Collection-2 L2 | USGS EarthExplorer | Public domain |
| Road network | OpenStreetMap via osmnx | ODbL |
| Healthcare facilities | OpenStreetMap | ODbL |
| Administrative boundary | GADM | Non-commercial |

---

## Dependencies

```
Python 3.11+
geopandas, rasterio, numpy, scipy
scikit-learn, tensorflow, keras
osmnx, matplotlib, pyproj, shapely
rasterstats
```

Install: `pip install -r requirements.txt`

---

## Citation

```
D. K., "Hospital Site Suitability Prediction Using CA-ANN and AHP-Based
Multi-Criteria Analysis for Vellore, Tamil Nadu," IEEE Conference, 2025.
```

---

*All code is open-source. Maps and rasters are reproducible from publicly available data.*
