"""Dash app: 7-day wildfire outlook (ignition, large fire, burn), who is exposed, and how hard
it is to get out."""
import base64
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from config import PROC, REPORTS, ROOT

FC = ROOT / "data" / "forecast"
TARGET_NAMES = {"ign": "Any fire starts", "large": "Large fire (100+ acres) starts", "fire": "Cell burns"}
LABELS = {
    "tmax_max": "Hot week (max temp)", "tmax_mean": "Hot week (avg temp)",
    "rhmin_min": "Very dry air (min humidity)", "rhmin_mean": "Dry air (avg humidity)",
    "wind_max": "Strong wind", "wind_mean": "Sustained wind", "vpd_max": "Peak drying power (VPD)",
    "vpd_mean": "Drying power (VPD)", "pr_week": "Rain this week", "pr_30d": "Rain past 30 days",
    "pr_90d": "Rain past 90 days", "pr_365d": "Rain past year", "days_since_rain": "Days since rain",
    "fm100_prev": "Dead fuel moisture (100-hr)", "fm1000_prev": "Dead fuel moisture (1000-hr)",
    "erc_prev": "Energy release component", "bi_prev": "Burning index",
    "erc_7d_prev": "ERC, past week", "erc_pct_prev": "ERC percentile",
    "offshore_days": "Offshore wind days", "santa_ana_days": "Santa Ana days",
    "santa_ana_wind_max": "Santa Ana wind speed",
    "elev_mean": "Elevation", "elev_std": "Rugged terrain", "slope_mean": "Steep slopes",
    "northness": "Slope aspect", "frac_tree": "Tree cover", "frac_shrub": "Shrub/chaparral cover",
    "frac_grass": "Grass cover", "frac_crop": "Cropland", "frac_built": "Built-up land",
    "frac_bare": "Bare ground", "log_pop_density": "People nearby",
    "road_km_per_km2": "Road density", "dist_major_road_km": "Distance to major road",
    "power_km_per_km2": "Power-line density", "dist_power_line_km": "Distance to power line",
    "dist_campground_km": "Distance to campground", "dist_border_km": "Distance to border",
    "dist_coast_km": "Distance to coast", "ring_veg": "Vegetation around cell",
    "ring_frac_shrub": "Chaparral around cell", "ring_frac_built": "Development around cell",
    "ring_housing_density": "Homes around cell", "wui_intermix": "WUI intermix",
    "wui_interface": "WUI interface", "log_housing_density": "Housing density",
    "ndvi": "Vegetation greenness", "ndvi_anom": "Greenness vs normal",
    "years_since_burn": "Years since last fire", "prior_burns": "Times burned since 1950",
    "perim_10km_20y": "Large fires within 10 km (20 yr)", "ign_cell_10y": "Fires started here (10 yr)",
    "ign_10km_30d": "Fires within 10 km (past 30 days)", "ign_10km_365d": "Fires within 10 km (past year)",
    "ign_10km_10y": "Fires within 10 km (10 yr)", "large_10km_10y": "Large fires within 10 km (10 yr)",
    "woy_sin": "Time of year", "woy_cos": "Time of year",
}

cells = gpd.read_parquet(PROC / "cells.parquet")
fc = pd.read_parquet(FC / "cells.parquet")
comm = pd.read_parquet(FC / "communities.parquet")
meta = json.load(open(FC / "meta.json"))
metrics = json.load(open(REPORTS / "metrics.json"))
monitoring = json.load(open(REPORTS / "monitoring.json")) if (REPORTS / "monitoring.json").exists() else {}

df = cells[["h3", "geometry"]].merge(fc, on="h3")
for t in TARGET_NAMES:
    df[f"pct_{t}"] = df[f"score_{t}"].rank(pct=True) * 100
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
    if feature.startswith("frac_") or feature.startswith("ring_frac") or feature == "ring_veg":
        return f"{v:.0%}"
    if feature.startswith("tmax"):
        return f"{v:.0f} °C"
    if feature.startswith("rhmin") or feature.startswith("fm"):
        return f"{v:.0f}%"
    if feature.startswith("vpd"):
        return f"{v:.1f} kPa"
    if feature.startswith("wind") or feature == "santa_ana_wind_max":
        return f"{v:.1f} m/s"
    if feature == "erc_pct_prev":
        return f"{v:.0%} of normal days are lower"
    if feature.startswith("dist_"):
        return f"{v:.1f} km"
    if feature.startswith("log_"):
        return f"{np.expm1(v):,.0f} per km²"
    if feature in ("woy_sin", "woy_cos"):
        return "this season"
    if feature.endswith("_days") or feature.startswith(("ign_", "large_", "perim_", "prior_")):
        return f"{v:.0f}"
    return f"{v:,.2f}"


def kpi(label, value):
    return html.Div([html.Div(value, className="kpi-v"), html.Div(label, className="kpi-l")], className="kpi")


def risk_map(target, layer):
    col = {"pct": f"pct_{target}", "p": f"p_{target}", "sd": f"sd_{target}", "exp": "exp_people"}[layer]
    title = {"pct": "Risk percentile", "p": "Probability (7 days)", "sd": "Model uncertainty",
             "exp": "Expected people exposed"}[layer]
    fig = go.Figure(go.Choroplethmap(
        geojson=geojson, locations=df.h3, z=df[col], colorscale="YlOrRd", marker_line_width=0,
        marker_opacity=0.75, colorbar=dict(title=dict(text=title, side="right"), thickness=12),
        customdata=np.stack([df.p_ign * 100, df.p_large * 100, df.p_fire * 100, df["pop"].fillna(0)], axis=1),
        hovertemplate="Any fire %{customdata[0]:.2f}%<br>Large fire %{customdata[1]:.3f}%<br>"
                      "Burns %{customdata[2]:.2f}%<br>Pop %{customdata[3]:,.0f}<extra></extra>"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=33.03, lon=-116.75), zoom=8.2),
                      margin=dict(l=0, r=0, t=0, b=0), height=560, clickmode="event")
    return fig


comm_table = comm.assign(
    exp_people=comm.exp_people.round(1), max_p_ign=(comm.max_p_ign * 100).round(2),
    max_p_large=(comm.max_p_large * 100).round(3), minutes_to_highway=comm.minutes_to_highway.round(1),
    people_per_route=comm.people_per_route.round(0),
    escape_routes=comm.escape_routes.map(lambda r: "—" if r != r else "10+" if r >= 10 else f"{r:.0f}"),
)[["NAME", "type", "pop", "max_p_ign", "max_p_large", "exp_people", "escape_routes", "minutes_to_highway",
   "people_per_route"]].sort_values("exp_people", ascending=False)


def results_table(target):
    res = pd.DataFrame(metrics["targets"][target]["results"]).T.reset_index(names="Method")
    res["pr_auc_lift"] = res.pr_auc_lift.round(1).astype(str) + "×"
    for c in ["roc_auc", "recall@5%", "recall@10%", "precision@5%"]:
        res[c] = res[c].round(3)
    return res[["Method", "pr_auc_lift", "roc_auc", "recall@5%", "recall@10%", "precision@5%"]]


def perf_section(target):
    m = metrics["targets"][target]
    rf = m.get("red_flag")
    parts = [html.H3(TARGET_NAMES[target]),
             dash_table.DataTable(results_table(target).to_dict("records"), style_cell={"fontFamily": "inherit"})]
    if rf:
        parts.append(html.P(
            f"Red Flag Warnings flagged {rf['share_flagged']:.1%} of cell-weeks and caught "
            f"{rf['rfw_recall']:.0%} of events; the model flagging the same number caught "
            f"{rf['model_recall_same_volume']:.0%}."))
    figs = [f"recall_{target}.png", f"shap_{target}.png", f"calibration_{target}.png"]
    if target == "fire":
        figs.append("backtest_valley.png")
    parts.append(html.Div([html.Img(src=img(REPORTS / p), className="fig") for p in figs
                           if (REPORTS / p).exists()], className="figs"))
    return html.Div(parts)


ign_m = metrics["targets"]["ign"]["results"]
live = monitoring.get("pooled_last_90", {}).get("ign_recall@10%")
app = Dash(__name__, title="SD Wildfire Risk")
server = app.server  # for gunicorn
app.layout = html.Div([
    html.H1("San Diego County Wildfire Outlook"),
    html.P(f"7-day outlook {meta['valid_from']} → {meta['valid_to']} · issued {meta['issued']} · "
           f"observed weather through {meta['observed_weather_through']}"
           + (f" · ⚠ {'; '.join(meta['warnings'])}" if meta.get("warnings") else ""), className="sub"),
    html.Div([
        kpi("Expected new fires this week (county)", f"{df.p_ign.sum():.1f}"),
        kpi("Expected people in burned area", f"{df.exp_people.sum():,.0f}"),
        kpi("Communities with ≤2 escape routes", f"{(comm.escape_routes <= 2).sum()}"),
        kpi("Fires caught by top-10% alerts (holdout)", f"{ign_m['Model']['recall@10%']:.0%}"
            + (f" · live {live:.0%}" if live is not None else "")),
    ], className="kpis"),
    dcc.Tabs([
        dcc.Tab(label="Risk map", children=[
            html.Div([
                dcc.RadioItems(id="target", options=[{"label": v, "value": k} for k, v in TARGET_NAMES.items()],
                               value="ign", inline=True, className="radio"),
                dcc.RadioItems(id="layer", options=[{"label": "Percentile", "value": "pct"},
                                                    {"label": "Probability", "value": "p"},
                                                    {"label": "Uncertainty", "value": "sd"},
                                                    {"label": "People exposed", "value": "exp"}],
                               value="pct", inline=True, className="radio"),
            ]),
            html.Div([
                dcc.Graph(id="map", figure=risk_map("ign", "pct"), className="map"),
                html.Div(id="detail", className="detail", children=html.P("Click a cell to see why it is risky.")),
            ], className="row"),
            html.H3("Communities ranked by expected people exposed"),
            dash_table.DataTable(
                comm_table.to_dict("records"), page_size=12, sort_action="native", filter_action="native",
                columns=[{"name": n, "id": i} for i, n in [
                    ("NAME", "Community"), ("type", "Type"), ("pop", "Population"),
                    ("max_p_ign", "Peak P(any fire) %"), ("max_p_large", "Peak P(large fire) %"),
                    ("exp_people", "Expected people exposed"), ("escape_routes", "Escape routes"),
                    ("minutes_to_highway", "Min. to highway"), ("people_per_route", "People per route")]],
                style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "6px"},
                style_header={"fontWeight": 600}),
        ]),
        dcc.Tab(label="Model performance", children=[
            html.P(f"All numbers are from a holdout the models never saw ({metrics['holdout']}); models were "
                   "trained on 2001–2018. Lift = PR-AUC ÷ base rate. recall@k = share of events in that week's "
                   "top-k% riskiest cells; precision@k = share of flagged cells that had an event."),
            *[perf_section(t) for t in TARGET_NAMES],
        ]),
    ]),
], className="page")


@app.callback(Output("map", "figure"), Input("target", "value"), Input("layer", "value"),
              prevent_initial_call=True)
def update_map(target, layer):
    return risk_map(target, layer)


@app.callback(Output("detail", "children"), Input("map", "clickData"), Input("target", "value"))
def detail(click, target):
    if not click:
        return html.P("Click a cell to see why it is risky.")
    r = df.set_index("h3").loc[click["points"][0]["location"]]
    drivers = [f"{LABELS.get(d, d)}: {fmt(d, r[d])}" for d in r[f"drivers_{target}"].split(", ")]
    return [
        html.H3(f"{r[f'p_{target}']:.2%}: {TARGET_NAMES[target].lower()}"),
        html.P(f"± {r[f'sd_{target}']:.2%} across models · {ordinal(round(r[f'pct_{target}']))} percentile"),
        html.P(f"Any fire {r.p_ign:.2%} · large fire {r.p_large:.3%} · burns {r.p_fire:.2%}"),
        html.P(f"Population: {r['pop']:,.0f} · Homes: {r.housing:,.0f}"),
        html.H4("Top drivers"), html.Ol([html.Li(d) for d in drivers]),
        html.H4("This week"),
        html.Ul([html.Li(f"Max temp {r.tmax_max:.0f} °C · min humidity {r.rhmin_min:.0f}%"),
                 html.Li(f"Peak wind {r.wind_max:.1f} m/s · Santa Ana days {r.santa_ana_days:.0f}"),
                 html.Li(f"ERC at {r.erc_pct_prev:.0%} percentile · {r.days_since_rain:.0f} days since rain"),
                 html.Li(f"{r.years_since_burn:.1f} years since last fire"
                         if r.years_since_burn < 75 else "No recorded fire since 1950")]),
    ]


if __name__ == "__main__":
    app.run(debug=False, port=8050)
