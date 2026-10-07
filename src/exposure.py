"""Who is exposed: 2020 Census population per cell and per community, plus
road-network evacuation metrics (escape routes, drive time to a highway)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests

from config import COUNTY_FIPS, CRS_M, PROC, RAW, STATE_FIPS

BLOCKS_URL = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{STATE_FIPS}_tabblock20.zip"
PLACES_URL = f"https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_{STATE_FIPS}_place.zip"
HIGHWAYS = {"motorway", "trunk", "motorway_link", "trunk_link"}
MAX_ROUTES = 10      # stop counting escape routes beyond this
SAFE_BUFFER_M = 2000  # a highway counts as "safety" only this far outside the community


def download(url: str) -> Path:
    path = RAW / url.rsplit("/", 1)[-1]
    if not path.exists():
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return path


def census_blocks() -> gpd.GeoDataFrame:
    blocks = gpd.read_file(download(BLOCKS_URL), where=f"COUNTYFP20 = '{COUNTY_FIPS}'",
                           columns=["GEOID20", "COUNTYFP20", "POP20", "HOUSING20"])
    blocks = blocks[blocks.POP20 > 0].to_crs(CRS_M)
    blocks["geometry"] = blocks.representative_point()
    return blocks.rename(columns={"POP20": "pop", "HOUSING20": "housing"})


def communities(county: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Incorporated cities + census-designated places inside the county."""
    places = gpd.read_file(download(PLACES_URL), columns=["GEOID", "NAME", "LSAD"]).to_crs(CRS_M)
    inside = places.representative_point().within(county.to_crs(CRS_M).geometry.iloc[0])
    places = places[inside].reset_index(drop=True)
    pop = gpd.sjoin(blocks, places[["GEOID", "geometry"]], predicate="within") \
        .groupby("GEOID")[["pop", "housing"]].sum()
    places = places.join(pop, on="GEOID").fillna({"pop": 0, "housing": 0})
    places["type"] = np.where(places.LSAD == "57", "CDP", "City")
    return places[places["pop"] > 0].drop(columns="LSAD").reset_index(drop=True)


def cell_population(cells: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame,
                    places: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Population per cell, and per (community, cell) for community-level expected impact."""
    cells_m = cells[["h3", "geometry"]].to_crs(CRS_M)
    b = gpd.sjoin(blocks, cells_m, predicate="within").drop(columns="index_right")
    cell_pop = b.groupby("h3")[["pop", "housing"]].sum().reindex(cells.h3, fill_value=0).reset_index()
    b = gpd.sjoin(b, places[["GEOID", "geometry"]], predicate="within")
    place_cell = b.groupby(["GEOID", "h3"])[["pop", "housing"]].sum().reset_index()
    return cell_pop, place_cell


def road_graph(county: gpd.GeoDataFrame) -> nx.Graph:
    ox.settings.requests_timeout = 600
    G = ox.graph_from_polygon(county.geometry.iloc[0], network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return ox.project_graph(G, to_crs=CRS_M)


def evacuation(G, places: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each community:
    - escape_routes: edge-disjoint road paths from inside the community to a highway at
      least SAFE_BUFFER_M outside it (max-flow / min-cut with unit capacities).
    - minutes_to_highway: median drive time from the community's road nodes to any highway."""
    U = nx.Graph(ox.convert.to_undirected(G))
    nodes = ox.convert.graph_to_gdfs(G, edges=False)[["geometry"]]
    hw_nodes = {n for u, v, d in U.edges(data=True)
                if HIGHWAYS & set(np.atleast_1d(d.get("highway"))) for n in (u, v)}
    nx.set_edge_attributes(U, 1, "cap")

    dist = nx.multi_source_dijkstra_path_length(U, hw_nodes, weight="travel_time")
    node_min = pd.Series(dist) / 60

    rows = []
    for _, p in places.iterrows():
        inside = nodes.index[nodes.within(p.geometry)]
        far = nodes.index[~nodes.within(p.geometry.buffer(SAFE_BUFFER_M))]
        sinks = hw_nodes.intersection(far)
        routes = np.nan
        if len(inside) and sinks:
            U.add_edges_from((("S", n) for n in inside), cap=MAX_ROUTES + 1)
            U.add_edges_from(((n, "T") for n in sinks), cap=MAX_ROUTES + 1)
            routes = nx.maximum_flow_value(U, "S", "T", capacity="cap",
                                           flow_func=nx.algorithms.flow.shortest_augmenting_path,
                                           cutoff=MAX_ROUTES)
            U.remove_nodes_from(["S", "T"])
        rows.append({"GEOID": p.GEOID, "road_nodes": len(inside),
                     "escape_routes": min(routes, MAX_ROUTES) if routes == routes else np.nan,
                     "minutes_to_highway": node_min.reindex(inside).median()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    county = gpd.read_parquet(PROC / "county.parquet")
    cells = gpd.read_parquet(PROC / "cells.parquet")
    blocks = census_blocks()
    places = communities(county, blocks)
    cell_pop, place_cell = cell_population(cells, blocks, places)
    cell_pop.to_parquet(PROC / "cell_pop.parquet")
    place_cell.to_parquet(PROC / "place_cell.parquet")
    print(f"{len(places)} communities, county pop {cell_pop['pop'].sum():,.0f}")

    G = road_graph(county)
    print(f"road graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    evac = evacuation(G, places)
    places = places.merge(evac, on="GEOID")
    places["people_per_route"] = (places["pop"] / places.escape_routes).where(places.escape_routes < MAX_ROUTES)
    places.to_crs(4326).to_parquet(PROC / "communities.parquet")
    print(places.sort_values("people_per_route", ascending=False)
          [["NAME", "type", "pop", "escape_routes", "minutes_to_highway", "people_per_route"]]
          .head(15).round(1).to_string(index=False))
