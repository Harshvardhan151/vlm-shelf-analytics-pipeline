import json
import os
import traceback
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Custom modules (analytical logic — unchanged)
from src.vlm_extractor import analyze_shelf_image
from src.compliance import calculate_compliance
from src.data_fusion import load_or_generate_sales_data, fuse_shelf_audit_with_sales
from src.forecasting import train_and_forecast

# UI layer (presentation only)
from ui import charts, components as ui
from ui.theme import inject_css

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(layout="wide", page_title="ShelfIQ | Retail Vision-to-Sales Analytics",
                   initial_sidebar_state="expanded")
inject_css()

if not api_key:
    st.error("Missing Gemini API Key. Please check your .env file.")
    st.stop()

# Session state so results survive reruns (tab switches, SKU selection, etc.)
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = {}

# Load Planogram for Aisle Navigation
with open('planogram.json', 'r') as f:
    planogram = json.load(f)

aisles = list(planogram.keys())

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    ui.sidebar_brand()

    ui.section_label("Aisle")
    selected_aisle = st.selectbox(
        "Store section", aisles, label_visibility="collapsed",
        help="Choose the store section you are auditing. The planogram for this aisle is used for comparison.",
    )
    # Extract expected products for the selected aisle
    expected_items = [item['brand_name'] for item in planogram[selected_aisle]]

    ui.section_label("Audit configuration")
    ui.aisle_summary(len(expected_items), expected_items)

    ui.sidebar_divider()
    ui.section_label("Workflow")
    ui.workflow_guide()

    if st.session_state.analysis_complete:
        ui.sidebar_divider()
        if st.button("Clear results", use_container_width=True,
                     help="Remove the current analysis and start a new audit."):
            for key in ("analysis_complete", "df", "comp_score", "foreign_objects",
                        "fused_df", "raw_sales_df", "expected_items", "analyzed_aisle"):
                st.session_state.pop(key, None)
            st.session_state.analysis_complete = False
            st.session_state.forecast_cache = {}
            st.rerun()

# ==========================================
# 3. HEADER & SHELF AUDIT WORKSPACE
# ==========================================
ui.page_header()

with st.container(border=True):
    ui.heading("Shelf Audit", f"Upload a photo of the {selected_aisle} to compare it against the planogram.")
    uploaded_file = st.file_uploader(
        f"Upload an image of the {selected_aisle}",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Supported formats: JPG, JPEG, PNG. Use a straight-on, well-lit photo of the full shelf.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG")

    if uploaded_file:
        col_img, col_info = st.columns([3, 2], gap="large")
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded {selected_aisle} State", use_container_width=True)

        with col_info:
            st.markdown(f"**Aisle:** {selected_aisle}")
            st.markdown(f"**Expected SKUs:** {len(expected_items)}")
            st.caption("Run the AI vision and business intelligence pipeline: shelf extraction, "
                       "planogram comparison, revenue impact, and historical data fusion.")

            if st.button("Run Shelf Analysis", type="primary", use_container_width=True):
                with st.spinner("AI is auditing the shelf and generating financial insights..."):
                    try:
                        # RUN PIPELINE (Extract -> Compare)
                        extracted_data = analyze_shelf_image(image, api_key, expected_items, selected_aisle)
                        detected_items = extracted_data["items"]

                        # Detect Foreign Objects
                        detected_brands = [item["brand_name"].strip().lower() for item in detected_items]
                        expected_brands_lower = [e.lower() for e in expected_items]
                        foreign_objects = [item["brand_name"] for item in detected_items if item["brand_name"].strip().lower() not in expected_brands_lower]

                        compliance_results = calculate_compliance(detected_items, planogram_path="planogram.json", aisle_name=selected_aisle)
                        df = compliance_results["comparison_df"]

                        # Apply mock dynamic pricing and sort by financial impact
                        price_map = {"Coca-Cola": 2.50, "Diet Coke": 2.50, "Sprite": 2.50, "Fanta": 2.50, "Doritos": 4.50, "Lays": 4.50, "Colgate Toothpaste": 6.50, "Dove Soap": 5.00}
                        df["unit_price"] = df["brand_name"].map(price_map).fillna(3.00)
                        df["lost_revenue_impact"] = df["facing_gap"].apply(lambda x: max(0, x)) * df["unit_price"]
                        df = df.sort_values(by="lost_revenue_impact", ascending=False)

                        # Fuse historical data
                        raw_sales_df = load_or_generate_sales_data()
                        fused_df = fuse_shelf_audit_with_sales(raw_sales_df, df, compliance_results["overall_compliance_score"])

                        # Save results to session state
                        st.session_state.analysis_complete = True
                        st.session_state.df = df
                        st.session_state.comp_score = compliance_results["overall_compliance_score"]
                        st.session_state.foreign_objects = foreign_objects
                        st.session_state.fused_df = fused_df
                        st.session_state.raw_sales_df = raw_sales_df
                        st.session_state.expected_items = expected_items
                        st.session_state.analyzed_aisle = selected_aisle
                        st.session_state.forecast_cache = {}  # new audit -> fresh forecasts

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        with st.expander("Technical details"):
                            st.code(traceback.format_exc())
                        st.session_state.analysis_complete = False

# ==========================================
# 4. ANALYTICS DASHBOARD (rendered from session state)
# ==========================================
ui.spacer()

if not st.session_state.analysis_complete:
    ui.empty_state(
        "No analysis yet",
        "Upload a shelf photo above and run the analysis to see compliance, revenue impact, "
        "historical trends and demand forecasts.",
    )
    st.stop()

df = st.session_state.df
comp_score = st.session_state.comp_score
foreign_objects = st.session_state.foreign_objects
fused_df = st.session_state.fused_df
raw_sales_df = st.session_state.raw_sales_df
aisle_expected_items = st.session_state.expected_items

# ---- KPI row ----
total_expected = df["expected_facings"].sum()
total_detected = df["detected_facings"].sum()
total_lost_revenue_daily = df["lost_revenue_impact"].sum()
facing_delta = int(total_detected - total_expected)

k1, k2, k3 = st.columns(3, gap="medium")
with k1:
    ui.kpi_card(
        "Planogram Compliance Score", f"{comp_score}%",
        "Perfect" if comp_score == 100 else f"-{round(100 - comp_score, 1)}% from target",
        tone="success" if comp_score >= 90 else "warning" if comp_score >= 70 else "critical",
    )
with k2:
    ui.kpi_card(
        "Total Detected Facings", str(total_detected),
        f"{facing_delta:+d} vs planogram",
        tone="success" if facing_delta == 0 else "warning",
    )
with k3:
    ui.kpi_card(
        "Estimated Daily Revenue at Risk", f"${total_lost_revenue_daily:.2f}",
        f"-${total_lost_revenue_daily:.2f}",
        tone="critical" if total_lost_revenue_daily > 0 else "success",
    )

ui.spacer()

tab_audit, tab_history, tab_forecast = st.tabs([
    "Shelf Audit & Compliance",
    "Historical Analytics",
    "Demand Forecasting & Inventory Planning",
])

# ---------------------------------------------------------
# TAB 1: SHELF AUDIT & COMPLIANCE
# ---------------------------------------------------------
with tab_audit:
    ui.spacer()
    ui.heading("Actionable Alerts", "Prioritized by revenue impact")

    for obj in foreign_objects:
        ui.alert(
            "warning", f"Misplaced item: {obj}",
            escape(f"'{obj}' was found here but belongs in a different aisle. Please remove."),
        )

    stockouts = df[df["has_stockout"] == True]
    if not stockouts.empty:
        for _, row in stockouts.iterrows():
            ui.alert(
                "critical", f"Restock: {row['brand_name']}",
                escape(f"Missing {row['facing_gap']} facings "
                       f"(detected {row['detected_facings']} of {row['expected_facings']} expected)"),
                impact=f"-${row['lost_revenue_impact']:.2f}", impact_label="daily revenue at risk",
            )
    else:
        ui.alert("success", "Shelf is fully compliant", "No restock needed.")

    ui.spacer()
    with st.container(border=True):
        ui.heading("Lost Revenue Waterfall", "Estimated daily revenue lost to missing facings, by SKU")
        st.plotly_chart(charts.waterfall(df), use_container_width=True)

    ui.spacer()
    ui.heading("Planogram Compliance Table", "Stockout rows are highlighted")

    def highlight_stockouts(row):
        if row["has_stockout"]:
            return ['background-color: #FEF2F2; color: #0F172A'] * len(row)
        return [''] * len(row)

    money_cols = {c: "${:,.2f}" for c in ("unit_price", "lost_revenue_impact") if c in df.columns}
    st.dataframe(
        df.style.apply(highlight_stockouts, axis=1).format(money_cols),
        use_container_width=True,
    )
    st.download_button(
        "Download table as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"shelfiq_compliance_{st.session_state.get('analyzed_aisle', 'aisle')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# TAB 2: HISTORICAL ANALYTICS
# ---------------------------------------------------------
with tab_history:
    ui.spacer()
    
    col_trend, col_lost_rev = st.columns([1, 1], gap="medium")
    with col_trend:
        with st.container(border=True):
            ui.heading("30-Day Store Compliance Trend", "Average visual shelf health across the store")
            daily_comp = raw_sales_df.groupby("date")["compliance_score"].mean().reset_index()
            st.plotly_chart(charts.compliance_trend(daily_comp), use_container_width=True)
            
    with col_lost_rev:
        with st.container(border=True):
            ui.heading("Estimated Lost Revenue Impact by SKU", "Revenue lost to missing product facings over the timeline")
            # Create price map and calculate lost volume & revenue
            price_map = {"Coca-Cola": 2.50, "Diet Coke": 2.50, "Sprite": 2.50, "Fanta": 2.50, "Doritos": 4.50, "Lays": 4.50, "Colgate Toothpaste": 6.50, "Dove Soap": 5.00}
            raw_sales_df["price"] = raw_sales_df["brand_name"].map(price_map).fillna(3.00)
            raw_sales_df["lost_volume"] = (raw_sales_df["baseline_demand"] - raw_sales_df["actual_sales"]).clip(lower=0)
            raw_sales_df["lost_revenue"] = raw_sales_df["lost_volume"] * raw_sales_df["price"]
            
            # Aggregate historical loss metrics
            sku_losses = raw_sales_df.groupby("brand_name")["lost_revenue"].sum().reset_index().sort_values(by="lost_revenue", ascending=True)
            sku_losses.columns = ["SKU", "Lost Revenue ($)"]
            
            import plotly.express as px
            fig_bar = px.bar(
                sku_losses, x="Lost Revenue ($)", y="SKU", orientation="h",
                text="Lost Revenue ($)", color="Lost Revenue ($)",
                color_continuous_scale="OrRd"
            )
            fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=40, t=10, b=10), height=310, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

    ui.spacer()
    
    col_heatmap, col_demand = st.columns([1, 1], gap="medium")
    with col_heatmap:
        with st.container(border=True):
            ui.heading("Vulnerability Heatmap", "Historical stockouts by weekday and SKU")
            raw_sales_df['day_of_week'] = raw_sales_df['date'].dt.day_name()
            heatmap_data = raw_sales_df[raw_sales_df['has_stockout'] == True].groupby(['day_of_week', 'brand_name']).size().reset_index(name='stockout_count')
            if not heatmap_data.empty:
                st.plotly_chart(charts.stockout_heatmap(heatmap_data), use_container_width=True)
            else:
                st.info("Not enough historical stockout data to generate a heatmap.")
                
    with col_demand:
        with st.container(border=True):
            ui.heading("Actual vs. Baseline Demand over Time", "Volume impact of demand censorship & shelf non-compliance")
            agg_timeline = raw_sales_df.groupby("date")[["actual_sales", "baseline_demand"]].sum().reset_index()
            agg_timeline["lost_sales_gap"] = (agg_timeline["baseline_demand"] - agg_timeline["actual_sales"]).clip(lower=0)
            
            import plotly.graph_objects as go
            fig_area = go.Figure()
            # Bottom Layer: Actual Sales (blue/cyan)
            fig_area.add_trace(go.Scatter(
                x=agg_timeline["date"], y=agg_timeline["actual_sales"],
                mode="lines", name="Actual Sales",
                fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.4)",
                line=dict(color="#1f77b4", width=1.5)
            ))
            # Stacked Layer: Gaps / Missed Demand (Crimson Red)
            fig_area.add_trace(go.Scatter(
                x=agg_timeline["date"], y=agg_timeline["baseline_demand"],
                mode="lines", name="Missed Demand (Gaps)",
                fill="tonexty", fillcolor="rgba(214, 39, 40, 0.45)",
                line=dict(color="#d62728", width=1.5)
            ))
            fig_area.update_layout(
                yaxis_title="Volume Demand (Units)", hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10), height=315,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_area, use_container_width=True)

    ui.spacer()
    with st.expander("View historical dataset"):
        st.dataframe(raw_sales_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: DEMAND FORECASTING & INVENTORY PLANNING
# ---------------------------------------------------------
with tab_forecast:
    ui.spacer()
    ui.heading("AI Forecasting & Capacity Planning",
               "Select a SKU to view its demand history, ML forecast and replenishment need")

    selected_brand = st.selectbox(
        "Select a specific SKU to generate ML Forecast:", aisle_expected_items,
        help="The XGBoost model is trained once per SKU per audit and cached for this session.",
    )

    brand_history = fused_df[fused_df["brand_name"] == selected_brand].sort_values("date")
    chart_view = brand_history.set_index("date")[["baseline_demand", "actual_sales"]]
    chart_view.columns = ["True Customer Demand (Uncensored)", "Recorded Register Sales"]

    # Training is unchanged; the result is reused when the same SKU is revisited
    # within the same audit so tab/widget reruns don't retrain the model.
    cache = st.session_state.forecast_cache
    if selected_brand not in cache:
        with st.spinner(f"Training XGBoost forecasting model for {selected_brand}..."):
            cache[selected_brand] = train_and_forecast(fused_df, selected_brand, forecast_days=7)
    forecast_df = cache[selected_brand]

    # --- NEW LOGIC: Volumetric Capacity & Turnover ---
    ASSUMED_SHELF_DEPTH = 5  # Assume shelves hold items 5 deep
    front_facings = df.loc[df['brand_name'] == selected_brand, 'expected_facings'].values[0] if not df[df['brand_name'] == selected_brand].empty else 1
    total_shelf_capacity = front_facings * ASSUMED_SHELF_DEPTH
    
    avg_daily_forecast = forecast_df['forecasted_demand'].mean()
    total_7_day_demand = forecast_df["forecasted_demand"].sum()
    
    # Calculate how many times the shelf will empty completely this week
    weekly_restocks_needed = total_7_day_demand / total_shelf_capacity if total_shelf_capacity > 0 else 0

    ui.spacer()
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        ui.kpi_card("Total Shelf Capacity (Volumetric)", f"{total_shelf_capacity} Units")
    with f2:
        ui.kpi_card("7-Day Predicted Demand", f"{int(total_7_day_demand)} Units")
    with f3:
        high_labor = weekly_restocks_needed > 3
        ui.kpi_card("Required Weekly Restocks", f"{weekly_restocks_needed:.1f}x",
                    "High Labor Required" if high_labor else "Normal Labor",
                    tone="critical" if high_labor else "success")

    ui.callout(
        f"<b>Logistics Alert:</b> The {escape(str(selected_brand))} shelf will need to be refilled approximately "
        f"{int(weekly_restocks_needed)} times over the next 7 days to keep up with forecasted demand."
    )

    with st.container(border=True):
        ui.heading(f"Sales Velocity vs. Shelf Shock Event: {selected_brand}",
                   "Recorded register sales versus uncensored customer demand")
        st.plotly_chart(charts.demand_vs_sales(chart_view), use_container_width=True)

    ui.spacer()
    
    # --- NEW LOGIC: Weekend vs Weekday Performance ---
    with st.container(border=True):
        ui.heading("Average Daily Sales: Weekend vs Weekday", f"Sales velocity pattern for {selected_brand}")
        
        # Calculate averages for weekdays (False) and weekends (True)
        if "is_weekend" in brand_history.columns:
            weekend_performance = brand_history.groupby("is_weekend")["actual_sales"].mean().reset_index()
            # Map boolean to string labels for the chart
            weekend_performance["is_weekend"] = weekend_performance["is_weekend"].map({True: "Weekend", False: "Weekday"})
            weekend_performance.columns = ["Day Type", "Avg Units Sold"]
            
            # Simple bar chart
            st.bar_chart(weekend_performance.set_index("Day Type"))
        else:
             st.info("Weekend indicator not found in historical data.")


    ui.spacer()
    col_cap, col_tbl = st.columns([3, 2], gap="medium")
    with col_cap:
        with st.container(border=True):
            ui.heading("7-Day Forecast vs. Total Shelf Capacity", f"Capacity planning: {selected_brand}")
            capacity_df = pd.DataFrame({
                "Metric": ["Total Shelf Capacity", "7-Day Total Demand"],
                "Units": [total_shelf_capacity, total_7_day_demand]
            })
            st.plotly_chart(charts.capacity_bar(capacity_df), use_container_width=True)
    with col_tbl:
        with st.container(border=True):
            ui.heading("7-Day Forecast Output", "Predicted daily demand")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    past_plot = brand_history[["date", "actual_sales"]].rename(columns={"actual_sales": "Sales/Forecast"})
    past_plot["Type"] = "Historical Sales"
    future_plot = forecast_df[["date", "forecasted_demand"]].rename(columns={"forecasted_demand": "Sales/Forecast"})
    future_plot["Type"] = "ML Forecast (Next 7 Days)"
    
    # --- NEW LOGIC: Fix the Chart Gap ---
    # Extract the very last day of historical sales
    last_historical_row = past_plot.iloc[[-1]].copy()
    # Change its Type to 'ML Forecast' so it acts as the starting anchor point for the forecast line
    last_historical_row["Type"] = "ML Forecast (Next 7 Days)"
    
    # Combine the data, injecting the anchor point between past and future
    combined_plot = pd.concat([past_plot, last_historical_row, future_plot]).set_index("date")

    # Force numeric casting to prevent mixed-type errors
    combined_plot["Sales/Forecast"] = pd.to_numeric(combined_plot["Sales/Forecast"])

    ui.spacer()
    with st.container(border=True):
        ui.heading(f"{selected_brand}: Historical Sales vs AI Forecast",
                   "Recorded sales followed by the 7-day ML forecast")
        st.plotly_chart(charts.history_vs_forecast(combined_plot), use_container_width=True)