from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image

# ==========================================
# 1. DEFINE THE IDEAL DATA STRUCTURE
# ==========================================
# This acts as our database schema. Gemini will map the image pixels directly into these fields.

class ShelfItem(BaseModel):
    brand_name: str = Field(description="Name of the product brand detected.")
    facings_count: int = Field(description="Number of identical items visible in the front row.")
    has_stockout: bool = Field(description="True if there is an empty gap/hole next to this product.")
    shelf_level: str = Field(description="Which shelf tier? (e.g., Top, Middle, Bottom).")

class ShelfReport(BaseModel):
    items: list[ShelfItem] = Field(description="List of all products detected on the shelf.")
    shelf_messiness_flag: bool = Field(description="True if products are knocked over, turned, or disorganized.")

# ==========================================
# 2. THE EXTRACTION ENGINE
# ==========================================

def analyze_shelf_image(image: Image.Image, api_key: str) -> dict:
    """
    Sends a shelf image to Gemini and returns structured data as a dictionary
    ready to be loaded into Pandas.
    """
    # Initialize the client with the key passed from app.py
    client = genai.Client(api_key=api_key)
    
    # The prompt sets the rules for the AI
    prompt = """
    You are a highly accurate retail shelf auditor. 
    Analyze this image and extract every visible product.
    Count the 'facings' (front-row items) for each brand.
    Identify which shelf level (Top, Middle, Bottom) they are on.
    Look carefully for empty spaces that indicate a stock-out gap.
    """
    
    # Call the model and enforce the JSON schema
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ShelfReport,
            temperature=0.1, # Extremely low temperature so it doesn't hallucinate data
        ),
    )
    
    # response.parsed contains the data mapped to our Pydantic classes
    report_data = response.parsed
    
    # Convert the Pydantic objects into a list of dictionaries for easy Pandas DataFrame creation
    return {
        "items": [item.model_dump() for item in report_data.items],
        "shelf_messiness_flag": report_data.shelf_messiness_flag
    }