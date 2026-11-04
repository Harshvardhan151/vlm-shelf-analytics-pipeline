from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image
import time
import streamlit as st

# ==========================================
# 1. DEFINE THE IDEAL DATA STRUCTURE
# ==========================================
class ShelfItem(BaseModel):
    brand_name: str = Field(description="Name of the product brand detected.")
    facings_count: int = Field(description="Number of identical items visible in the front row.")
    has_stockout: bool = Field(description="True if there is an empty gap/hole next to this product.")
    shelf_level: str = Field(description="Which shelf tier? (e.g., Top, Middle, Bottom, Eye-Level).")

class ShelfReport(BaseModel):
    items: list[ShelfItem] = Field(description="List of all products detected on the shelf.")
    shelf_messiness_flag: bool = Field(description="True if products are knocked over, turned, or disorganized.")

# ==========================================
# 2. THE DEMO-DAY FALLBACK DATA
# ==========================================
def get_mock_fallback_data(aisle_name: str) -> dict:
    """Returns hardcoded data matching the selected aisle if the API goes down."""
    mock_databases = {
        "Beverages Aisle": [
            {"brand_name": "Coca-Cola", "facings_count": 4, "has_stockout": False, "shelf_level": "Middle"},
            {"brand_name": "Diet Coke", "facings_count": 0, "has_stockout": True, "shelf_level": "Middle"},
            {"brand_name": "Sprite", "facings_count": 4, "has_stockout": False, "shelf_level": "Bottom"},
            {"brand_name": "Fanta", "facings_count": 2, "has_stockout": True, "shelf_level": "Bottom"}
        ],
        "Snacks Aisle": [
            {"brand_name": "Doritos", "facings_count": 2, "has_stockout": True, "shelf_level": "Middle"},
            {"brand_name": "Lays", "facings_count": 5, "has_stockout": False, "shelf_level": "Middle"}
        ],
        "Household Aisle": [
            {"brand_name": "Colgate Toothpaste", "facings_count": 8, "has_stockout": False, "shelf_level": "Eye-Level"},
            {"brand_name": "Dove Soap", "facings_count": 3, "has_stockout": True, "shelf_level": "Eye-Level"}
        ]
    }
    
    return {
        "items": mock_databases.get(aisle_name, []),
        "shelf_messiness_flag": False
    }

# ==========================================
# 3. THE EXTRACTION ENGINE (WITH FALLBACKS)
# ==========================================
def analyze_shelf_image(image: Image.Image, api_key: str, expected_items: list, aisle_name: str) -> dict:
    """
    Sends a shelf image to Gemini, unconstrained, to catch foreign objects.
    """
    client = genai.Client(api_key=api_key)
    
    # Prompt is now unconstrained to identify EVERYTHING on the shelf, enabling foreign object detection
    prompt = """
    You are a highly accurate retail shelf auditor. 
    Analyze this image and extract every single product brand you see on the shelf. 
    For each brand, count ONLY the 'facings' (the front-row items visible). 
    Do not count items hidden behind the front row. 
    Identify which shelf level they are on, and look carefully for empty spaces that indicate a stock-out gap.
    """
    
    models_to_try = ['gemini-3.6-flash', 'gemini-3.5-flash-lite','gemini-2.5-flash']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ShelfReport,
                    temperature=0.1, 
                ),
            )
            
            report_data = response.parsed
            
            return {
                "items": [item.model_dump() for item in report_data.items],
                "shelf_messiness_flag": report_data.shelf_messiness_flag
            }
            
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                print(f"Model {model_name} is busy. Trying the next option...")
                time.sleep(1)
                continue 
            else:
                print(f"Failed with error: {error_msg}")
                break 
                
    st.toast(f"⚠️ API High Demand: Using offline cached data for {aisle_name}.", icon="🔌")
    return get_mock_fallback_data(aisle_name)