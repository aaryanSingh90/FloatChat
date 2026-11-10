"""
visualization/charts.py
========================
Plotly chart builders for the Float Chat Streamlit dashboard.
All functions return a Plotly Figure that can be rendered with
st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (ocean-inspired)
# ---------------------------------------------------------------------------
OCEAN_PALETTE = [
    "#0077B6", "#00B4D8", "#90E0EF", "#CAF0F8",
    "#023E8A", "#0096C7", "#48CAE4", "#ADE8F4",
]
TEMP_CMAP = "RdBu_r"
SAL_CMAP = "Viridis"


# ---------------------------------------------------------------------------
# 1. Geographic map of float positions
# ---------------------------------------------------------------------------

def float_map(df: pd.DataFrame) -> go.Figure:
    """Scatter map of float observation positions coloured by temperature."""
    if df.empty:
        return _empty_fig("No data to display on map.")

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="temperature",
        color_continuous_scale=TEMP_CMAP,
        hover_data={
            "float_id": True,
            "region": True,
            "date": True,
            "temperature": ":.2f",
            "salinity": ":.2f",
            "depth": ":.1f",
            "qc_flag": True,
        },
        title="🌊 Float Observation Positions",
        mapbox_style="carto-positron",
        zoom=1,
        height=500,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Temp (°C)"),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Temperature vs depth profile
# ---------------------------------------------------------------------------

def temperature_depth_profile(df: pd.DataFrame, region: str | None = None) -> go.Figure:
    """Scatter plot of temperature vs depth, optionally filtered by region."""
    data = df if region is None else df[df["region"] == region]
    if data.empty:
        return _empty_fig("No data for selected region.")

    fig = px.scatter(
        data,
        x="temperature",
        y="depth",
        color="region",
        color_discrete_sequence=OCEAN_PALETTE,
        hover_data={"date": True, "float_id": True, "salinity": ":.2f"},
        title="🌡️ Temperature vs Depth Profile",
        labels={"temperature": "Temperature (°C)", "depth": "Depth (m)"},
    )
    fig.update_yaxes(autorange="reversed", title="Depth (m) ↓")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        legend_title_text="Region",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Salinity vs depth profile
# ---------------------------------------------------------------------------

def salinity_depth_profile(df: pd.DataFrame, region: str | None = None) -> go.Figure:
    """Scatter plot of salinity vs depth."""
    data = df if region is None else df[df["region"] == region]
    if data.empty:
        return _empty_fig("No data for selected region.")

    fig = px.scatter(
        data,
        x="salinity",
        y="depth",
        color="region",
        color_discrete_sequence=OCEAN_PALETTE,
        hover_data={"date": True, "float_id": True, "temperature": ":.2f"},
        title="🧂 Salinity vs Depth Profile",
        labels={"salinity": "Salinity (PSU)", "depth": "Depth (m)"},
    )
    fig.update_yaxes(autorange="reversed", title="Depth (m) ↓")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        legend_title_text="Region",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Temperature-Salinity diagram
# ---------------------------------------------------------------------------

def ts_diagram(df: pd.DataFrame) -> go.Figure:
    """Classic oceanographic T-S diagram coloured by region."""
    if df.empty:
        return _empty_fig("No data for T-S diagram.")

    fig = px.scatter(
        df,
        x="salinity",
        y="temperature",
        color="region",
        color_discrete_sequence=OCEAN_PALETTE,
        hover_data={"depth": ":.1f", "date": True, "float_id": True},
        title="🔬 Temperature–Salinity (T-S) Diagram",
        labels={"salinity": "Salinity (PSU)", "temperature": "Temperature (°C)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        legend_title_text="Region",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Regional statistics bar chart
# ---------------------------------------------------------------------------

def regional_stats_bar(df: pd.DataFrame, variable: str = "temperature") -> go.Figure:
    """Bar chart of mean, min, max per region for *variable*."""
    if df.empty or variable not in df.columns:
        return _empty_fig(f"No data for {variable}.")

    stats = df.groupby("region")[variable].agg(["mean", "min", "max"]).reset_index()
    stats.columns = ["region", "Mean", "Min", "Max"]

    fig = go.Figure()
    for col, colour in zip(["Mean", "Min", "Max"], ["#0077B6", "#00B4D8", "#90E0EF"]):
        fig.add_trace(go.Bar(
            name=col,
            x=stats["region"],
            y=stats[col],
            marker_color=colour,
        ))

    fig.update_layout(
        barmode="group",
        title=f"📊 {variable.capitalize()} Statistics by Region",
        xaxis_title="Region",
        yaxis_title=f"{variable.capitalize()} (°C)" if variable == "temperature" else variable,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        legend_title_text="Statistic",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. QC flag distribution
# ---------------------------------------------------------------------------

def qc_flag_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing distribution of QC flags."""
    if df.empty:
        return _empty_fig("No data.")

    counts = df["qc_flag"].value_counts().reset_index()
    counts.columns = ["qc_flag", "count"]
    flag_labels = {1: "1 – Good", 2: "2 – Prob. Good", 3: "3 – Prob. Bad", 4: "4 – Bad"}
    counts["label"] = counts["qc_flag"].map(flag_labels).fillna(counts["qc_flag"].astype(str))

    fig = px.pie(
        counts,
        names="label",
        values="count",
        title="🏷️ QC Flag Distribution",
        color_discrete_sequence=["#0077B6", "#00B4D8", "#FF9900", "#FF3300"],
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
    return fig


# ---------------------------------------------------------------------------
# 7. Time-series line chart
# ---------------------------------------------------------------------------

def time_series(df: pd.DataFrame, variable: str = "temperature") -> go.Figure:
    """Line chart of *variable* over time, coloured by region."""
    if df.empty or variable not in df.columns:
        return _empty_fig(f"No data for {variable}.")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    fig = px.line(
        data,
        x="date",
        y=variable,
        color="region",
        color_discrete_sequence=OCEAN_PALETTE,
        title=f"📈 {variable.capitalize()} Over Time",
        labels={"date": "Date", variable: f"{variable.capitalize()}"},
        markers=True,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        legend_title_text="Region",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Query result bar chart (generic)
# ---------------------------------------------------------------------------

def query_result_bar(result_df: pd.DataFrame, title: str = "Query Results") -> go.Figure:
    """Auto-detect x/y columns and render a bar chart from a query result."""
    if result_df.empty:
        return _empty_fig("No results to display.")

    numeric_cols = result_df.select_dtypes("number").columns.tolist()
    cat_cols = result_df.select_dtypes(["object", "category"]).columns.tolist()

    if not numeric_cols:
        return _empty_fig("No numeric data to plot.")

    x_col = cat_cols[0] if cat_cols else result_df.columns[0]
    y_col = numeric_cols[0]

    fig = px.bar(
        result_df,
        x=x_col,
        y=y_col,
        color=x_col if len(result_df) <= 20 else None,
        color_discrete_sequence=OCEAN_PALETTE,
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,248,255,0.8)",
        showlegend=False,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#555"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig

