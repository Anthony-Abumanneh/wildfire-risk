"""Dash app: 7-day wildfire risk, who is exposed, and how hard it is to get out."""
import base64
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from config import PROC, REPORTS, ROOT

FC = ROOT / "data" / "forecast"
LABELS = {
    "tmax_max": "Hot week (max temp)", "tmax_mean": "Hot week (avg temp)",
    "rhmin_min": "Very dry air (min humidity)", "rhmin_mean": "Dry air (avg humidity)",
    "wind_max": "Strong wind", "wind_mean": "Sustained wind", "vpd_max": "Peak drying power (VPD)",
    "vpd_mean": "Drying power (VPD)", "pr_week": "Rain this week", "pr_30d": "Rain past 30 days",
    "pr_90d": "Rain past 90 days", "pr_365d": "Rain past year", "days_since_rain": "Days since rain",
    "elev_mean": "Elevation", "elev_std": "Rugged terrain", "slope_mean": "Steep slopes",
    "northness": "Slope aspect", "frac_tree": "Tree cover", "frac_shrub": "Shrub/chaparral cover",
    "frac_grass": "Grass cover", "frac_crop": "Cropland", "frac_built": "Built-up land",
    "frac_bare": "Bare ground", "log_pop_density": "People nearby (ignitions)",
    "ndvi": "Vegetation greenness", "ndvi_anom": "Greenness vs normal",
    "years_since_burn": "Years since last fire", "prior_burns": "Fire history",
    "woy_sin": "Time of year", "woy_cos": "Time of year",
}
LAYERS = {"pct": "Risk percentile", "p_fire": "Fire probability (7 days)",
          "exp_people": "Expected people exposed", "p_sd": "Model uncertainty"}

cells = gpd.read_parquet(PROC / "cells.parquet")
fc = pd.read_parquet(FC / "cells.parquet")
comm = pd.read_parquet(FC / "communities.parquet")
meta = json.load(open(FC / "meta.json"))
metrics = json.load(open(REPORTS / "metrics.json"))

df = cells[["h3", "geometry"]].merge(fc, on="h3")
df["pct"] = df.score.rank(pct=True) * 100
geojson = json.loads(df[["h3", "geometry"]].to_json())
for f in geojson["features"]:
    f["id"] = f["properties"]["h3"]


def img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt(feature: str, v: float) -> str:
    if feature.startswith("pr_"):
        return f"{v:.0f} mm"
    if feature.startswith("frac_"):
        return f"{v:.0%}"
    if feature.startswith("tmax"):
        return f"{v:.0f} °C"
    if feature.startswith("rhmin"):
        return f"{v:.0f}%"
    if feature.startswith("vpd"):
        return f"{v:.1f} kPa"
    if feature.startswith("wind"):
        return f"{v:.1f} m/s"
    if feature == "log_pop_density":
        return f"{np.expm1(v):,.0f} people/km²"
    if feature in ("woy_sin", "woy_cos"):
        return "this season"
    return f"{v:,.2f}"


def kpi(label, value):
    return html.Div([html.Div(value, className="kpi-v"), html.Div(label, className="kpi-l")],
                    className="kpi")


def risk_map(layer):
    z = df[layer]
    fig = go.Figure(go.Choroplethmap(
        geojson=geojson, locations=df.h3, z=z, colorscale="YlOrRd", marker_line_width=0,
        marker_opacity=0.75, colorbar=dict(title=dict(text=LAYERS[layer], side="right"), thickness=12),
        customdata=np.stack([df.p_fire * 100, df.p_sd * 100, df["pop"].fillna(0)], axis=1),
        hovertemplate="P(fire) %{customdata[0]:.2f}% ± %{customdata[1]:.2f}<br>"
                      "Pop %{customdata[2]:,.0f}<extra></extra>"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=33.03, lon=-116.75), zoom=8.2),
                      margin=dict(l=0, r=0, t=0, b=0), height=560, clickmode="event")
    return fig


comm_table = comm.assign(
    exp_people=comm.exp_people.round(1), max_cell_p=(comm.max_cell_p * 100).round(2),
    minutes_to_highway=comm.minutes_to_highway.round(1),
    people_per_route=comm.people_per_route.round(0),
    escape_routes=comm.escape_routes.map(lambda r: "—" if r != r else "10+" if r >= 10 else f"{r:.0f}"),
)[["NAME", "type", "pop", "max_cell_p", "exp_people", "escape_routes", "minutes_to_highway",
   "people_per_route"]].sort_values("exp_people", ascending=False)

res = pd.DataFrame(metrics["results"]).T.reset_index(names="Model")
for c in ["pr_auc", "roc_auc", "capture_top5", "capture_top10"]:
    res[c] = res[c].round(3)
res["pr_auc_lift"] = res.pr_auc_lift.round(1).astype(str) + "×"
res = res[["Model", "pr_auc", "pr_auc_lift", "roc_auc", "capture_top5", "capture_top10"]]

app = Dash(__name__, title="SD Wildfire Risk")
app.layout = html.Div([
    html.H1("San Diego County Wildfire Risk"),
    html.P(f"7-day outlook {meta['valid_from']} → {meta['valid_to']} · issued {meta['issued']} · "
           f"observed weather through {meta['observed_weather_through']}", className="sub"),
    html.Div([
        kpi("Highest cell fire probability", f"{df.p_fire.max():.2%}"),
        kpi("Expected people in burned area", f"{df.exp_people.sum():,.0f}"),
        kpi("Communities with ≤2 escape routes", f"{(comm.escape_routes <= 2).sum()}"),
        kpi("Holdout PR-AUC vs. climatology",
            f"{metrics['results']['Full model - temporal holdout (2019+)']['pr_auc'] / metrics['results']['Climatology baseline - temporal holdout']['pr_auc']:.1f}×"),
    ], className="kpis"),
    dcc.Tabs([
        dcc.Tab(label="Risk map", children=[
            dcc.RadioItems(id="layer", options=[{"label": v, "value": k} for k, v in LAYERS.items()],
                           value="pct", inline=True, className="radio"),
            html.Div([
                dcc.Graph(id="map", figure=risk_map("pct"), className="map"),
                html.Div(id="detail", className="detail",
                         children=html.P("Click a cell to see why it is risky.")),
            ], className="row"),
            html.H3("Communities ranked by expected people exposed"),
            dash_table.DataTable(
                comm_table.to_dict("records"), page_size=12, sort_action="native",
                filter_action="native",
                columns=[{"name": n, "id": i} for i, n in [
                    ("NAME", "Community"), ("type", "Type"), ("pop", "Population"),
                    ("max_cell_p", "Peak P(fire) %"), ("exp_people", "Expected people exposed"),
                    ("escape_routes", "Escape routes"), ("minutes_to_highway", "Min. to highway"),
                    ("people_per_route", "People per route")]],
                style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "6px"},
                style_header={"fontWeight": 600}),
        ]),
        dcc.Tab(label="Model performance", children=[
            html.P("Temporal holdout = models trained on 2001–2018, scored on 2019+. PR-AUC lift = PR-AUC "
                   "÷ base fire rate. Capture@k = share of burned cells that were in the top k% riskiest "
                   "cells that week."),
            dash_table.DataTable(res.to_dict("records"), style_cell={"fontFamily": "inherit"}),
            html.Div([html.Img(src=img(REPORTS / p), className="fig") for p in
                      ["shap_importance.png", "calibration.png", "backtest_valley.png"]
                      if (REPORTS / p).exists()], className="figs"),
        ]),
    ]),
], className="page")


@app.callback(Output("map", "figure"), Input("layer", "value"), prevent_initial_call=True)
def update_map(layer):
    return risk_map(layer)


@app.callback(Output("detail", "children"), Input("map", "clickData"))
def detail(click):
    if not click:
        return html.P("Click a cell to see why it is risky.")
    r = df.set_index("h3").loc[click["points"][0]["location"]]
    drivers = [f"{LABELS.get(d, d)}: {fmt(d, r[d])}" for d in r.drivers.split(", ")]
    return [
        html.H3(f"{r.p_fire:.2%} chance of fire"),
        html.P(f"± {r.p_sd:.2%} across models · {ordinal(round(r.pct))} percentile in the county"),
        html.P(f"Population: {r['pop']:,.0f} · Homes: {r.housing:,.0f}"),
        html.H4("Top drivers"), html.Ol([html.Li(d) for d in drivers]),
        html.H4("This week"),
        html.Ul([html.Li(f"Max temp {r.tmax_max:.0f} °C"), html.Li(f"Min humidity {r.rhmin_min:.0f}%"),
                 html.Li(f"Peak wind {r.wind_max:.1f} m/s"),
                 html.Li(f"{r.days_since_rain:.0f} days since rain"),
                 html.Li(f"{r.years_since_burn:.1f} years since last fire"
                         if r.years_since_burn < 75 else "No recorded fire since 1950")]),
    ]


if __name__ == "__main__":
    app.run(debug=False, port=8050)
