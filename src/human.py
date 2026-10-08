"""Human ignition drivers and neighborhood context per cell (static).

- OSM roads (length density, distance to a major road), power lines, campgrounds
- Distance to the US-Mexico border and to the coast (Natural Earth)
- SILVIS-style wildland-urban interface: intermix / interface from housing density + vegetation
- Ring-1 neighborhood averages of land cover and population
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd

from config import CRS_M, PROC

NE = "https://naciscdn.org/naturalearth/10m/{}.zip"
MAJOR = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link"}
WUI_HOUSING = 6.17     # housing units / km² (SILVIS threshold)
WUI_VEG = 0.5          # intermix: wildland vegetation share
INTERFACE_VEG = 0.75   # interface: near (ring-1, ~2.4 km) a heavily vegetated cell
CELL_KM2 = 5.16


def length_per_cell(lines: gpd.GeoDataFrame, cells_m: gpd.GeoDataFrame) -> pd.Series:
    """Line length (km) inside each cell / cell area -> km per km²."""
    parts = gpd.overlay(lines[["geometry"]], cells_m[["h3", "geometry"]], how="intersection",
                        keep_geom_type=True)
    return (parts.length.groupby(parts.h3).sum() / 1000 / CELL_KM2)


def nearest_km(ctr: gpd.GeoDataFrame, target: gpd.GeoDataFrame) -> np.ndarray:
    j = gpd.sjoin_nearest(ctr[["h3", "geometry"]], target[["geometry"]], distance_col="d")
    return j.groupby("h3").d.min().reindex(ctr.h3).to_numpy() / 1000


OVERPASS_MIRRORS = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api",
                    "https://overpass.private.coffee/api"]


def features_with_fallback(poly, tags):
    for url in OVERPASS_MIRRORS:
        ox.settings.overpass_url = url
        try:
            return ox.features_from_polygon(poly, tags=tags)
        except Exception as e:  # mirror down / refused -> try the next one
            print(f"  overpass {url} failed: {type(e).__name__}")
    raise RuntimeError("all Overpass mirrors failed")


def osm_features(county, cells_m, ctr) -> pd.DataFrame:
    poly = county.geometry.iloc[0]
    G = ox.graph_from_polygon(poly, network_type="drive")  # cached from exposure.py
    edges = ox.convert.graph_to_gdfs(G, nodes=False)[["highway", "geometry"]].to_crs(CRS_M)
    major = edges[edges.highway.apply(lambda h: bool(MAJOR & set(np.atleast_1d(h))))]

    power = features_with_fallback(poly, {"power": ["line", "minor_line"]})
    power = power[power.geom_type.isin(["LineString", "MultiLineString"])].to_crs(CRS_M)
    camps = features_with_fallback(poly, {"tourism": ["camp_site", "caravan_site"]}).to_crs(CRS_M)
    camps = camps.set_geometry(camps.representative_point())

    out = pd.DataFrame({"h3": ctr.h3})
    out["road_km_per_km2"] = out.h3.map(length_per_cell(edges, cells_m)).fillna(0).to_numpy()
    out["dist_major_road_km"] = nearest_km(ctr, major)
    out["power_km_per_km2"] = out.h3.map(length_per_cell(power, cells_m)).fillna(0).to_numpy()
    out["dist_power_line_km"] = nearest_km(ctr, power)
    out["dist_campground_km"] = nearest_km(ctr, camps)
    print(f"  OSM: {len(edges):,} road edges, {len(power):,} power lines, {len(camps)} campgrounds")
    return out


def natural_earth_distances(ctr) -> pd.DataFrame:
    borders = gpd.read_file(NE.format("cultural/ne_10m_admin_0_boundary_lines_land"))
    us_mx = borders[borders.ADM0_LEFT.isin(["United States of America", "Mexico"]) &
                    borders.ADM0_RIGHT.isin(["United States of America", "Mexico"])].to_crs(CRS_M)
    coast = gpd.read_file(NE.format("physical/ne_10m_coastline")).to_crs(CRS_M)
    coast = coast.clip(ctr.total_bounds + np.array([-1e5, -1e5, 1e5, 1e5]))
    return pd.DataFrame({"h3": ctr.h3, "dist_border_km": nearest_km(ctr, us_mx),
                         "dist_coast_km": nearest_km(ctr, coast)})


def wui_and_neighbors(static: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    df = static.merge(pop, on="h3", how="left").fillna({"pop": 0, "housing": 0}).set_index("h3")
    df["veg"] = df.frac_tree + df.frac_shrub + df.frac_grass
    df["housing_density"] = df.housing / CELL_KM2
    ring = {c: [n for n in h3.grid_disk(c, 1) if n != c and n in df.index] for c in df.index}

    def ring_mean(col):
        return pd.Series({c: df.loc[nb, col].mean() if nb else df.loc[c, col] for c, nb in ring.items()})

    out = pd.DataFrame(index=df.index)
    for col in ["veg", "frac_shrub", "frac_built", "housing_density"]:
        out[f"ring_{col}"] = ring_mean(col)
    ring_veg_max = pd.Series({c: df.loc[nb, "veg"].max() if nb else 0 for c, nb in ring.items()})
    housed = df.housing_density > WUI_HOUSING
    out["wui_intermix"] = (housed & (df.veg > WUI_VEG)).astype(float)
    out["wui_interface"] = (housed & (df.veg <= WUI_VEG) & (ring_veg_max > INTERFACE_VEG)).astype(float)
    out["log_housing_density"] = np.log1p(df.housing_density)
    return out.reset_index(names="h3")


if __name__ == "__main__":
    county = gpd.read_parquet(PROC / "county.parquet")
    cells = gpd.read_parquet(PROC / "cells.parquet")
    cells_m = cells.to_crs(CRS_M)
    ctr = cells_m.set_geometry(cells_m.centroid)
    static = pd.read_parquet(PROC / "static.parquet")
    pop = pd.read_parquet(PROC / "cell_pop.parquet")

    human = osm_features(county, cells_m, ctr) \
        .merge(natural_earth_distances(ctr), on="h3") \
        .merge(wui_and_neighbors(static, pop), on="h3")
    human.to_parquet(PROC / "human.parquet")
    print(human.describe().T[["mean", "min", "max"]].round(2).to_string())
    print(f"WUI intermix cells: {int(human.wui_intermix.sum())}, interface: {int(human.wui_interface.sum())}")
