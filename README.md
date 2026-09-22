# 🛒 Retail Vision-to-Sales Pipeline: Smart Shelf Analytics

An end-to-end data pipeline that connects physical retail shelf images to business forecasting and financial impact analysis using Vision-Language Models (VLMs) and Machine Learning.

**Built for: University Data Analysis & Visualization Project**

## 📖 Project Overview
In the retail industry, **"Phantom Stock-Outs"** occur when a store's database says inventory exists, but the physical shelf is actually empty. This creates **Censored Demand**—sales drop to zero, and traditional forecasting models misinterpret this as a drop in customer interest, causing the system to under-order for the following week.

This project solves this by using a **Vision-Language Model (Gemini 3.6 Flash)** to instantly audit shelf photos, calculate planogram compliance, merge the visual data with historical point-of-sale (POS) data, and train an **XGBoost** machine learning model to forecast true future demand.

## ✨ Key Features (The 5 Phases)
1. **VLM Feature Extraction:** Replaces traditional heavy Computer Vision (like YOLO) with a VLM API. Uses **Pydantic** to force Gemini to output strictly structured JSON data (brand names, facing counts, stock-out gaps).
2. **Planogram Compliance Engine:** Compares the VLM's extracted data against an ideal layout (`planogram.json`) to generate a shelf health score.
3. **Data Fusion:** Mathematically fuses physical visual data with historical time-series sales data, simulating demand shocks when stock-outs occur.
4. **Machine Learning Forecasting:** Trains an **XGBoost** regression model on the fused data to predict true inventory needs for the next 7 days.
5. **Interactive Dashboard:** A fully interactive **Streamlit** web application that visualizes the pipeline, metrics, and forecasts.

## 🛠️ Tech Stack
* **UI & Dashboard:** Streamlit
* **AI / Vision:** Google GenAI API (`gemini-3.6-flash`)
* **Data Structuring:** Pydantic
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn

## 📂 Project Structure
```text
vlm-shelf-analytics-pipeline/
│
├── src/                   
│   ├── vlm_extractor.py   # Handles Gemini API calls & structured extraction
│   ├── compliance.py      # Calculates planogram matching & error scores
│   ├── data_fusion.py     # Merges visual data with time-series POS data
│   └── forecasting.py     # XGBoost ML model for demand forecasting
│
├── app.py                 # Main Streamlit dashboard application
├── planogram.json         # The "ideal" shelf layout target
├── requirements.txt       # Python dependencies
├── .env                   # (Ignored) Stores the Google Gemini API Key
└── .gitignore             # Prevents secrets and cache from being uploaded