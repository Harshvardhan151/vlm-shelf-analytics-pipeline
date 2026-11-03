"""Plotly figure builders. Each takes the exact data the original charts used."""
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import (BORDER, CRITICAL, NEUTRAL_SERIES, PRIMARY, style_figure)

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def waterfall(df):
    fig = go.Figure(go.Waterfall(
        name="Lost Revenue", orientation="v",
        measure=["relative"] * len(df),
        x=df["brand_name"],
        y=-df["lost_revenue_impact"],
        textposition="outside",
        text=df["lost_revenue_impact"].apply(lambda x: f"-${x:.2f}" if x > 0 else ""),
        decreasing=dict(marker=dict(color=CRITICAL)),
        increasing=dict(marker=dict(color=NEUTRAL_SERIES)),
        totals=dict(marker=dict(color=PRIMARY)),
        connector=dict(line=dict(color=BORDER, width=1)),
        hovertemplate="<b>%{x}</b><br>Revenue impact: %{y:$,.2f}<extra></extra>",
    ))
    fig.update_layout(waterfallgap=0.3)
    style_figure(fig, title="Financial Bleed by SKU", x_title="SKU",
                 y_title="Daily revenue impact (USD)", legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def compliance_trend(daily_comp):
    fig = px.line(daily_comp, x="date", y="compliance_score", markers=True)
    fig.update_traces(
        line=dict(color=PRIMARY, width=2.5), marker=dict(size=6, color=PRIMARY),
        hovertemplate="%{x|%b %d, %Y}<br>Compliance: %{y:.1f}%<extra></extra>",
    )
    return style_figure(fig, title="Visual Shelf Health Over Time", x_title="Date",
                        y_title="Compliance score (%)", legend=False)


def stockout_heatmap(heatmap_data):
    fig = px.density_heatmap(
        heatmap_data, x="day_of_week", y="brand_name", z="stockout_count",
        color_continuous_scale=["#F8FAFC", "#FECACA", "#F87171", CRITICAL],
        category_orders={"day_of_week": _DAY_ORDER},
    )
    fig.update_traces(
        xgap=3, ygap=3,
        hovertemplate="%{y} on %{x}<br>Stockouts: %{z}<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Stockouts", thickness=12, len=0.8))
    style_figure(fig, title="Stockout Frequency by Day and SKU", x_title="Day of week",
                 y_title="SKU", legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def demand_vs_sales(chart_view):
    fig = go.Figure()
    styles = [(PRIMARY, "solid"), (NEUTRAL_SERIES, "solid")]
    for col, (color, dash) in zip(chart_view.columns, styles):
        fig.add_trace(go.Scatter(
            x=chart_view.index, y=chart_view[col], name=col, mode="lines",
            line=dict(color=color, width=2.5, dash=dash),
            hovertemplate="%{y:.1f} units<extra>" + col + "</extra>",
        ))
    return style_figure(fig, x_title="Date", y_title="Units per day")


def capacity_bar(capacity_df):
    fig = px.bar(
        capacity_df, x="Metric", y="Units", color="Metric", text="Units",
        color_discrete_map={"Shelf Capacity": NEUTRAL_SERIES, "7-Day Total Forecast": PRIMARY},
    )
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside",
                      hovertemplate="%{x}: %{y:.1f} units<extra></extra>")
    fig.update_layout(bargap=0.45)
    style_figure(fig, x_title=None, y_title="Units", legend=False, height=340)
    fig.update_layout(hovermode="closest")
    return fig


def history_vs_forecast(combined_plot):
    data = combined_plot.reset_index()
    fig = px.line(
        data, x="date", y="Sales/Forecast", color="Type",
        color_discrete_map={"Historical Sales": NEUTRAL_SERIES, "ML Forecast (Next 7 Days)": PRIMARY},
    )
    fig.update_traces(line=dict(width=2.5), hovertemplate="%{y:.1f} units<extra>%{fullData.name}</extra>")
    fig.for_each_trace(lambda t: t.update(line=dict(dash="dash")) if t.name.startswith("ML Forecast") else ())
    return style_figure(fig, x_title="Date", y_title="Units per day")
