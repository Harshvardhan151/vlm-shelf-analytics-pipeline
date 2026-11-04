import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from src.vlm_extractor import analyze_shelf_image
from src.compliance import calculate_compliance
from src.data_fusion import load_or_generate_sales_data, fuse_shelf_audit_with_sales
from src.forecasting import train_and_forecast

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(layout="wide", page_title="Smart Shelf Analytics")
st.title("🛒 Retail Vision-to-Sales Pipeline")

if not api_key:
    st.error("Missing Gemini API Key. Please check your .env file.")
    st.stop()

# Initialize session state so graphs don't disappear when interacting with dropdowns
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# Load Planogram for Aisle Navigation
with open('planogram.json', 'r') as f:
    planogram = json.load(f)

# ==========================================
# 2. UI: AISLE SELECTION & IMAGE UPLOAD
# ==========================================
st.markdown("### Step 1: Select the Aisle")
aisles = list(planogram.keys())
selected_aisle = st.selectbox("Which store section are you auditing?", aisles)

# Extract expected products for the selected aisle
expected_items = [item['brand_name'] for item in planogram[selected_aisle]]

st.markdown("### Step 2: Upload Shelf Photo")
uploaded_file = st.file_uploader(f"Upload an image of the {selected_aisle}", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_info = st.columns([1, 2])
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {selected_aisle} State", use_container_width=True)

    with col_info:
        st.info("Click below to run the AI Vision and Business Intelligence pipeline.")
        
        if st.button("Run Planogram Analysis", type="primary"):
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

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.session_state.analysis_complete = False

# ==========================================
# 3. VISUALIZATION DASHBOARD (Rendered from Session State)
# ==========================================
if st.session_state.analysis_complete:
    df = st.session_state.df
    comp_score = st.session_state.comp_score
    foreign_objects = st.session_state.foreign_objects
    fused_df = st.session_state.fused_df
    raw_sales_df = st.session_state.raw_sales_df
    aisle_expected_items = st.session_state.expected_items

    st.divider()
    st.header("📊 Shelf Analytics & Financial Report")
    
    if foreign_objects:
        for obj in foreign_objects:
            st.warning(f"⚠️ **Misplaced Item Detected:** '{obj}' was found here but belongs in a different aisle. Please remove.")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Overall Compliance Score", f"{comp_score}%", delta="Perfect" if comp_score == 100 else f"-{round(100-comp_score, 1)}% from target")
    
    total_expected = df["expected_facings"].sum()
    total_detected = df["detected_facings"].sum()
    metric2.metric("Total Facings", total_detected, delta=int(total_detected - total_expected))
    
    total_lost_revenue_daily = df["lost_revenue_impact"].sum()
    metric3.metric("Est. Lost Revenue (Daily)", f"${total_lost_revenue_daily:.2f}", delta=f"-${total_lost_revenue_daily:.2f}", delta_color="inverse")

    st.subheader("Lost Revenue Waterfall")
    fig_waterfall = go.Figure(go.Waterfall(
        name="Lost Revenue", orientation="v",
        measure=["relative"] * len(df),
        x=df["brand_name"],
        y=-df["lost_revenue_impact"],
        textposition="outside",
        text=df["lost_revenue_impact"].apply(lambda x: f"-${x:.2f}" if x > 0 else "")
    ))
    fig_waterfall.update_layout(title="Financial Bleed by SKU", waterfallgap=0.3)
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.subheader("Actionable Alerts (Prioritized by Revenue Impact)")
    stockouts = df[df["has_stockout"] == True]
    if not stockouts.empty:
        for _, row in stockouts.iterrows():
            st.error(f"🚨 **Restock Alert (Priority)**: {row['brand_name']} is missing {row['facing_gap']} facings. **Impact: -${row['lost_revenue_impact']:.2f}**")
    else:
        st.success("✅ Shelf is fully compliant! No restock needed.")
        
    st.subheader("Planogram Compliance Table")
    def highlight_stockouts(row):
        if row["has_stockout"]:
            return ['background-color: #ffcccc; color: black'] * len(row)
        return [''] * len(row)
    st.dataframe(df.style.apply(highlight_stockouts, axis=1), use_container_width=True)

    st.divider()
    st.header("📈 Historical Trends & Data Fusion")
    
    st.subheader("30-Day Store Compliance Trend")
    daily_comp = raw_sales_df.groupby("date")["compliance_score"].mean().reset_index()
    fig_trend = px.line(daily_comp, x="date", y="compliance_score", markers=True, title="Visual Shelf Health Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader("Vulnerability Heatmap (Stockouts by Day)")
    raw_sales_df['day_of_week'] = raw_sales_df['date'].dt.day_name()
    heatmap_data = raw_sales_df[raw_sales_df['has_stockout'] == True].groupby(['day_of_week', 'brand_name']).size().reset_index(name='stockout_count')
    if not heatmap_data.empty:
        fig_heat = px.density_heatmap(heatmap_data, x="day_of_week", y="brand_name", z="stockout_count", color_continuous_scale="Reds")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough historical stockout data to generate a heatmap.")

    st.divider()
    st.header("🔮 Deep-Dive: AI Forecasting & Capacity Planning")
    
    # ---------------------------------------------------------
    # NEW: Interactive SKU Dropdown (Works without resetting app)
    # ---------------------------------------------------------
    selected_brand = st.selectbox("Select a specific SKU to generate ML Forecast:", aisle_expected_items)
    
    brand_history = fused_df[fused_df["brand_name"] == selected_brand].sort_values("date")
    chart_view = brand_history.set_index("date")[["baseline_demand", "actual_sales"]]
    chart_view.columns = ["True Customer Demand (Uncensored)", "Recorded Register Sales"]
    
    st.subheader(f"Sales Velocity vs. Shelf Shock Event: {selected_brand}")
    st.line_chart(chart_view)

    with st.spinner(f"Training XGBoost forecasting model for {selected_brand}..."):
        forecast_df = train_and_forecast(fused_df, selected_brand, forecast_days=7)
        
        current_inventory = df.loc[df['brand_name'] == selected_brand, 'detected_facings'].values[0] if not df[df['brand_name'] == selected_brand].empty else 0
        avg_daily_forecast = forecast_df['forecasted_demand'].mean()
        days_of_supply = current_inventory / avg_daily_forecast if avg_daily_forecast > 0 else 0
        
        total_needed = forecast_df["forecasted_demand"].sum()
        shelf_capacity = df.loc[df['brand_name'] == selected_brand, 'expected_facings'].values[0] if not df[df['brand_name'] == selected_brand].empty else 0
        
        kpi_f1, kpi_f2, kpi_f3 = st.columns(3)
        kpi_f1.metric("Current Visible Inventory", f"{current_inventory} Units")
        kpi_f2.metric("Avg Forecasted Daily Demand", f"{avg_daily_forecast:.1f} Units/Day")
        kpi_f3.metric("Days of Supply Remaining", f"{days_of_supply:.1f} Days", delta="Critical Alert" if days_of_supply < 2 else "Healthy", delta_color="inverse" if days_of_supply < 2 else "normal")
        
        st.success(f"**Action Required:** Order {int(total_needed)} units of {selected_brand} to cover expected demand over the next 7 days.")
        
        st.subheader("7-Day Forecast vs. Physical Shelf Capacity")
        capacity_df = pd.DataFrame({
            "Metric": ["Shelf Capacity", "7-Day Total Forecast"],
            "Units": [shelf_capacity, total_needed]
        })
        fig_capacity = px.bar(capacity_df, x="Metric", y="Units", color="Metric", title=f"Capacity Planning: {selected_brand}")
        st.plotly_chart(fig_capacity, use_container_width=True)
        
        past_plot = brand_history[["date", "actual_sales"]].rename(columns={"actual_sales": "Sales/Forecast"})
        past_plot["Type"] = "Historical Sales"
        future_plot = forecast_df[["date", "forecasted_demand"]].rename(columns={"forecasted_demand": "Sales/Forecast"})
        future_plot["Type"] = "ML Forecast (Next 7 Days)"
        
        combined_plot = pd.concat([past_plot, future_plot]).set_index("date")
        
        # Force numeric casting to prevent mixed-type Streamlit errors
        combined_plot["Sales/Forecast"] = pd.to_numeric(combined_plot["Sales/Forecast"])
        
        st.subheader(f"{selected_brand}: Historical Sales vs AI Forecast")
        # Explicitly define the y-axis
        st.line_chart(combined_plot, y="Sales/Forecast", color="Type")