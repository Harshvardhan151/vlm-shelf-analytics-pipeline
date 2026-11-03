ShelfIQ: VLM-Driven Shelf Analytics and Demand Forecasting

ShelfIQ is an AI-powered retail intelligence platform that fuses Vision-Language Models (VLMs) with traditional Machine Learning to automate retail shelf auditing and predict future inventory logistics.

By analyzing live photographs of physical store shelves, ShelfIQ compares real-world conditions against digital planograms to detect stockouts, flag misplaced items, and calculate exact revenue lost to poor visual merchandising. It then leverages this physical context to train a real-time ML model that predicts future customer demand and calculates the physical labor required to keep shelves stocked.

🚀 Core Capabilities

Zero-Shot Visual Auditing (VLM): Utilizes the Google Gemini API to extract brand names and visual facings from raw photographs without the need for custom object-detection training or bounding-box labeling.

Planogram Compliance Engine: Cross-references VLM observations against a strict set of digital rules (planogram.json) to instantly identify facing gaps and "foreign objects."

Financial Impact & Demand Censorship: Fuses visual compliance gaps with historical sales data to calculate "Lost Revenue" and simulates how missing units mathematically censor actual sales velocity.

Real-Time ML Forecasting: Trains a local XGBoost model on-the-fly to predict a 7-day sales forecast for specific SKUs, accounting for historical patterns like weekend traffic surges.

Logistics & Capacity Planning: Translates the AI forecast into physical labor metrics by calculating how many times a shelf will completely empty and require physical restocking per week based on its volumetric depth.

🧠 Technical Architecture

The project relies on a hybrid AI approach:

Cloud-Based Pre-Trained AI (Perception): The Gemini Vision model acts as the perceptual layer, translating unstructured pixels into structured JSON data.

Local Real-Time ML (Prediction): An XGBoost model acts as the analytical layer, training locally in real-time on tabular time-series data to predict future trends.

Directory Structure

shelfiq/
├── app.py                     # Streamlit frontend and UI dashboard
├── planogram.json             # The "Ground Truth" rules for aisle layouts
├── clean_retail_sales.csv     # 90-day static historical sales dataset
├── .env                       # Environment variables (API Key)
├── src/
│   ├── vlm_extractor.py       # Interfaces with Gemini API for image extraction
│   ├── compliance.py          # Calculates facing gaps and shelf health scores
│   ├── data_fusion.py         # Merges audit data with historical sales/dates
│   └── forecasting.py         # XGBoost model training and 7-day predictions
└── ui/                        # Custom charts and Streamlit CSS components


⚙️ Installation & Setup

Clone the Repository: Ensure the entire project directory is downloaded to your local machine.

Install Dependencies:
Install the required Python libraries using pip:

pip install streamlit pandas numpy plotly python-dotenv google-generativeai xgboost pillow


Configure API Key:
Create a .env file in the root directory (alongside app.py) and add your Google Gemini API key:

GEMINI_API_KEY="your_api_key_here"


Run the Application:
Start the Streamlit server:

streamlit run app.py


📊 Running a Demonstration (Viva Guide)

Select an Aisle: Use the sidebar to select an aisle (e.g., "Beverages").

Upload Evidence: Upload a photograph of a physical retail shelf. (For best results, use an image with clear brand visibility and obvious stockouts or misplaced items).

Run Analysis: Click "Run Shelf Analysis." The system will ping the VLM, calculate compliance, and update the historical database.

Review Dashboard:

Tab 1 (Shelf Audit): View the Lost Revenue Waterfall and actionable restock alerts.

Tab 2 (Historical Analytics): View the dynamic 30-day compliance trend and stockout heatmaps.

Tab 3 (Forecasting): Select a SKU to train the XGBoost model and view the calculated weekend vs. weekday performance and 7-day labor/capacity predictions.