"""
Assessment tools for evaluating supply chain situation criticality.
"""

from typing import Dict, List

def criticality_assessment_tool(
    drug_info: Dict,
    stockout_history: List[Dict],  # List of stockout events for this drug/region
    unfulfilled_demand: float,     # Total unfulfilled for this drug/region over history period
    total_demand: float            # Total demand for this drug/region over history period
) -> Dict:
    """
    Assesses the criticality of a drug supply situation based on stockout history
    and unfulfilled demand, providing recommendations for action.
    
    Args:
        drug_info: Information about the drug being assessed
        stockout_history: Record of stockout events
        unfulfilled_demand: Amount of unfulfilled demand
        total_demand: Total demand amount for comparison
        
    Returns:
        Dict: Assessment results including criticality score, category, and recommendations
    """
    drug_criticality = drug_info.get("criticality_value", 1)
    drug_name = drug_info.get("name", "Unknown")

    # Metrics based on provided history (assume history is recent, e.g., last 10 days)
    stockout_days = len(set(s['day'] for s in stockout_history)) # Count unique days with stockouts

    if total_demand > 0:
        unfulfilled_percentage = (unfulfilled_demand / total_demand) * 100
    else:
        unfulfilled_percentage = 0 if unfulfilled_demand == 0 else 100 # If no demand but unfulfilled somehow, treat as 100%

    # Criticality Score (0-100, adjusted scale)
    # Base score from drug itself (max 40)
    base_score = drug_criticality * 10
    # Penalty for stockout frequency (max 30) - more sensitive to recent stockouts
    stockout_penalty = min(30, stockout_days * 5) # Penalize each day of stockout more
    # Penalty for unfulfilled demand severity (max 30)
    unfulfilled_penalty = min(30, unfulfilled_percentage * 0.6) # Scale percentage penalty

    criticality_score = base_score + stockout_penalty + unfulfilled_penalty
    criticality_score = min(100, max(0, criticality_score)) # Cap between 0 and 100

    # Determine category
    if criticality_score >= 80: 
        category = "Critical Emergency"
    elif criticality_score >= 60: 
        category = "Severe Shortage"
    elif criticality_score >= 40: 
        category = "Moderate Concern"
    elif criticality_score >= 20: 
        category = "Potential Issue"
    else: 
        category = "Normal Operations"

    # Generate Recommendations (tailored to score)
    recommendations = []
    if category == "Critical Emergency":
        recommendations.extend([
            "PRIORITY 1: Request IMMEDIATE emergency allocation/resupply.",
            "Activate strict rationing protocols NOW.",
            "Urgently seek therapeutic alternatives.",
            "Escalate issue to regional/central command."
        ])
    elif category == "Severe Shortage":
        recommendations.extend([
            "Significantly increase order quantities (e.g., 2x-3x normal).",
            "Request expedited delivery of pending orders.",
            "Implement patient prioritization criteria.",
            "Notify regional coordinator of severe shortage."
        ])
    elif category == "Moderate Concern":
        recommendations.extend([
            "Increase safety stock targets.",
            "Place supplementary order (e.g., 1.5x normal).",
            "Review usage patterns for potential optimization.",
            "Monitor inbound shipments closely."
        ])
    elif category == "Potential Issue":
        recommendations.extend([
            "Monitor inventory and demand trends very closely.",
            "Consider a small increase in next order.",
            "Verify accuracy of demand forecast."
        ])
    else: # Normal Operations
        recommendations.append("Maintain standard ordering procedures based on forecast.")

    assessment = {
        "drug_name": drug_name,
        "criticality_score": round(criticality_score, 1),
        "category": category,
        "stockout_days_recent": stockout_days,
        "unfulfilled_percentage_recent": round(unfulfilled_percentage, 1),
        "recommendations": recommendations
    }
    return assessment