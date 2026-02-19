import osmnx as ox
import geopandas as gpd

# ── 1. Vellore boundary ──────────────────────────────────────────────
vellore = ox.geocode_to_gdf("Vellore, Tamil Nadu, India")
vellore.to_file("data/processed/vellore_boundary.gpkg", driver="GPKG")
print("✅ Boundary saved:", vellore.total_bounds)

# ── 2. Road network (new API) ────────────────────────────────────────
G = ox.graph_from_place("Vellore, Tamil Nadu, India", network_type="drive")
ox.save_graph_geopackage(G, filepath="data/processed/roads/vellore_roads.gpkg")

nodes, edges = ox.graph_to_gdfs(G)
print(f"✅ Roads saved: {len(edges)} edges, {len(nodes)} nodes")

# ── 3. Existing hospitals from OSM ──────────────────────────────────
hospitals = ox.features_from_place(
    "Vellore, Tamil Nadu, India",
    tags={"amenity": ["hospital", "clinic", "health_post"]}
)
hospitals_gdf = gpd.GeoDataFrame(hospitals[["geometry", "name", "amenity"]])
hospitals_gdf = hospitals_gdf[hospitals_gdf.geometry.geom_type == "Point"]
hospitals_gdf.to_file("data/processed/hospitals_osm.gpkg", driver="GPKG")
print(f"✅ Hospitals found: {len(hospitals_gdf)}")
print(hospitals_gdf[["name", "amenity"]].to_string())

# ── 4. Quick visual check ────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.5, color="gray", alpha=0.7)
vellore.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=2)
hospitals_gdf.plot(ax=ax, color="red", markersize=50, marker="+", label="Hospitals")
ax.set_title("Vellore — Roads + Existing Hospitals", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("maps/vellore_base_map.png", dpi=150)
plt.show()
print("✅ Base map saved to maps/vellore_base_map.png")