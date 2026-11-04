import json
import pandas as pd

# Embedded price map for financial impact calculations
PRICE_MAP = {
    "coca-cola": 2.50, 
    "diet coke": 2.50, 
    "sprite": 2.50, 
    "fanta": 2.50, 
    "doritos": 4.50, 
    "lays": 4.50, 
    "colgate toothpaste": 6.50, 
    "dove soap": 5.00
}

def load_planogram(file_path: str = "planogram.json") -> dict:
    """Loads the ideal planogram reference data."""
    with open(file_path, "r") as f:
        return json.load(f)

def calculate_compliance(detected_items: list[dict], planogram_path: str = "planogram.json", aisle_name: str = None) -> dict:
    """
    Compares VLM-detected shelf items with the target planogram.
    Returns an overall compliance score (0-100) and a comparison DataFrame.
    """
    planogram = load_planogram(planogram_path)
    
    # Extract the expected items dynamically based on the selected aisle
    if aisle_name and aisle_name in planogram:
        expected_items = planogram[aisle_name]
    else:
        expected_items = []

    # Index detected items by lowercase brand name for fuzzy matching
    detected_map = {item["brand_name"].strip().lower(): item for item in detected_items}

    comparison_records = []
    item_scores = []

    for expected in expected_items:
        exp_brand = expected["brand_name"]
        exp_facings = expected["expected_facings"]
        exp_level = expected["expected_shelf_level"]

        # Look up detected product
        matched = detected_map.get(exp_brand.strip().lower())

        if matched:
            det_facings = matched.get("facings_count", 0)
            det_level = matched.get("shelf_level", "Unknown")
            has_stockout = matched.get("has_stockout", False)

            # Sub-score 1: Facing ratio (max 100%)
            facing_ratio = min(1.0, det_facings / exp_facings) if exp_facings > 0 else 1.0

            # Sub-score 2: Position penalty REMOVED. We no longer care about exact shelf placement.
            
            # Sub-score 3: Stockout penalty
            stockout_penalty = 0.2 if has_stockout else 0.0

            # Combined item score (weighted: 80% facings, 20% shelf health)
            score = max(0.0, (facing_ratio * 0.8 + (1.0 - stockout_penalty) * 0.2)) * 100
            status = "Stocked" if not has_stockout else "Stockout Detected"
        else:
            # Completely missing from shelf
            det_facings = 0
            det_level = "Missing"
            has_stockout = True
            score = 0.0
            status = "Missing SKU"

        # Financial Impact Calculation
        facing_gap = exp_facings - det_facings
        unit_price = PRICE_MAP.get(exp_brand.strip().lower(), 3.00) # Default to $3.00 if unknown
        lost_revenue_impact = max(0, facing_gap) * unit_price

        item_scores.append(score)
        comparison_records.append({
            "brand_name": exp_brand,
            "expected_facings": exp_facings,
            "detected_facings": det_facings,
            "facing_gap": facing_gap,
            "expected_level": exp_level,
            "detected_level": det_level,
            "has_stockout": has_stockout,
            "status": status,
            "compliance_score": round(score, 1),
            "unit_price": unit_price,
            "lost_revenue_impact": lost_revenue_impact
        })

    # Overall shelf compliance is the mean score across all planogram target items
    overall_compliance = round(sum(item_scores) / len(item_scores), 1) if item_scores else 0.0

    return {
        "overall_compliance_score": overall_compliance,
        "comparison_df": pd.DataFrame(comparison_records)
    }