import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image

# Import the custom modules you built in the src/ folder
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

# ==========================================
# 2. UI: IMAGE UPLOAD
# ==========================================
st.write("Upload a shelf photo below to extract visual inventory and compare it to the planogram.")
uploaded_file = st.file_uploader("Upload a shelf photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    col_img, col_info = st.columns([1, 2])
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Shelf State", use_container_width=True)

    with col_info:
        st.info("Click the button below to send this image to the Vision-Language Model.")
        
        if st.button("Run Planogram Analysis", type="primary"):
            with st.spinner("AI is scanning the shelf and calculating demand gaps..."):
                
                # ==========================================
                # 3. RUN PIPELINE (Extract -> Compare)
                # ==========================================
                try:
                    # Step A: VLM Extraction
                    extracted_data = analyze_shelf_image(image, api_key)
                    detected_items = extracted_data["items"]
                    
                    # Step B: Planogram Comparison
                    compliance_results = calculate_compliance(detected_items, planogram_path="planogram.json")
                    
                    comp_score = compliance_results["overall_compliance_score"]
                    df = compliance_results["comparison_df"]
                    
                    # ==========================================
                    # 4. VISUALIZATION DASHBOARD
                    # ==========================================
                    st.divider()
                    st.header("📊 Shelf Analytics Report")
                    
                    # Top-Level Metrics
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric(
                        "Overall Compliance Score", 
                        f"{comp_score}%", 
                        delta="Perfect" if comp_score == 100 else f"-{round(100-comp_score, 1)}% from target"
                    )
                    
                    total_expected = df["expected_facings"].sum()
                    total_detected = df["detected_facings"].sum()
                    metric2.metric("Total Facings", total_detected, delta=int(total_detected - total_expected))
                    
                    # Mock calculation for lost revenue: $4.50 per missing facing
                    lost_revenue_daily = df["facing_gap"].apply(lambda x: max(0, x)).sum() * 4.50
                    metric3.metric("Est. Lost Revenue (Daily)", f"${lost_revenue_daily:.2f}", delta=f"-${lost_revenue_daily:.2f}", delta_color="inverse")

                    # Comparison Table with Color Highlighting
                    st.subheader("Planogram Compliance Table")
                    
                    # Function to highlight rows where there is a stockout
                    def highlight_stockouts(row):
                        if row["has_stockout"]:
                            return ['background-color: #ffcccc; color: black'] * len(row)
                        return [''] * len(row)
                        
                    st.dataframe(df.style.apply(highlight_stockouts, axis=1), use_container_width=True)
                    
                    # Bar Chart: Expected vs. Detected
                    st.subheader("Expected vs. Detected Facings")
                    # Reshape data slightly so Streamlit can plot multiple bars side-by-side
                    chart_data = df[["brand_name", "expected_facings", "detected_facings"]].set_index("brand_name")
                    st.bar_chart(chart_data)
                    
                    # Actionable Alerts Box
                    st.subheader("Actionable Alerts")
                    stockouts = df[df["has_stockout"] == True]
                    if not stockouts.empty:
                        for _, row in stockouts.iterrows():
                            st.error(f"🚨 **Restock Alert**: {row['brand_name']} is missing {row['facing_gap']} facings on the {row['expected_level']} shelf!")
                    else:
                        st.success("✅ Shelf is fully compliant! No restock needed.")

                    # ==========================================
                    # 5. DATA FUSION: POS SALES & DEMAND SHOCKS
                    # ==========================================
                    st.divider()
                    st.header("📈 Financial & Demand Fusion Analysis")
                    st.write(
                        "Connecting physical visual shelf state with daily point-of-sale data "
                        "to isolate true demand vs. lost sales caused by shelf gaps."
                    )

                    # Load historical sales and merge current audit
                    raw_sales_df = load_or_generate_sales_data()
                    fused_df = fuse_shelf_audit_with_sales(raw_sales_df, df, comp_score)

                    # Key Financial KPIs
                    audit_day_sales = fused_df[fused_df["date"] == fused_df["date"].max()]
                    total_actual = audit_day_sales["actual_sales"].sum()
                    total_potential = audit_day_sales["baseline_demand"].sum()
                    total_lost_dollars = audit_day_sales["lost_revenue"].sum()

                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Recorded Units Sold", total_actual)
                    kpi2.metric("True Potential Demand", total_potential, delta=int(total_potential - total_actual))
                    kpi3.metric("Stockout Revenue Loss", f"${total_lost_dollars:.2f}", delta=f"-${total_lost_dollars:.2f}", delta_color="inverse")

                    # Time Series Chart: True Demand vs. Recorded Sales
                    st.subheader("Sales Velocity vs. Shelf Shock Event")
                    
                    selected_brand = st.selectbox("Select SKU to view demand trend:", df["brand_name"].unique())
                    brand_history = fused_df[fused_df["brand_name"] == selected_brand].sort_values("date")

                    # Pivot for Streamlit multi-line plotting
                    chart_view = brand_history.set_index("date")[["baseline_demand", "actual_sales"]]
                    chart_view.columns = ["True Customer Demand (Uncensored)", "Recorded Register Sales"]
                    
                    st.line_chart(chart_view)

                    if audit_day_sales[audit_day_sales["brand_name"] == selected_brand]["has_stockout"].values[0]:
                        st.warning(
                            f"⚠️ **Demand Shock Identified:** On the audit date, {selected_brand} sales dropped "
                            f"strictly because the shelf was empty, not because customer interest decreased."
                        )

                    # ==========================================
                    # 6. PHASE 5: AI DEMAND FORECASTING (XGBoost)
                    # ==========================================
                    st.divider()
                    st.header("🔮 7-Day Restock Forecast")
                    st.write(
                        "Using an XGBoost Machine Learning model trained on the fused visual/sales data "
                        "to predict exactly how much inventory to order for next week."
                    )
                    
                    with st.spinner("Training XGBoost forecasting model..."):
                        # Train the model and get 7-day future predictions for the selected brand
                        forecast_df = train_and_forecast(fused_df, selected_brand, forecast_days=7)
                        
                        # Display the forecast metrics
                        total_needed = forecast_df["forecasted_demand"].sum()
                        st.success(f"**Action Required:** Order {int(total_needed)} units of {selected_brand} to cover demand for the next 7 days.")
                        
                        # Prepare data for plotting (combine past actuals with future forecast)
                        past_plot = brand_history[["date", "actual_sales"]].rename(columns={"actual_sales": "Sales/Forecast"})
                        past_plot["Type"] = "Historical Sales"
                        
                        future_plot = forecast_df[["date", "forecasted_demand"]].rename(columns={"forecasted_demand": "Sales/Forecast"})
                        future_plot["Type"] = "ML Forecast (Next 7 Days)"
                        
                        # Combine and plot
                        combined_plot = pd.concat([past_plot, future_plot]).set_index("date")
                        
                        st.subheader(f"{selected_brand} - Historical vs Forecast")
                        # Streamlit's native line chart handles this beautifully
                        st.line_chart(combined_plot, color="Type")
                        
                        # Show raw forecast data in an expander for the professor
                        with st.expander("View Raw ML Forecast Data"):
                            st.dataframe(forecast_df, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")