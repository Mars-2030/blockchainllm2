"""
Forecasting tools for pandemic supply chain simulation.
"""

import numpy as np
import math
from typing import List, Dict
from sklearn.linear_model import LinearRegression

def epidemic_forecast_tool(
    current_cases: float, case_history: List[float], days_to_forecast: int = 14
) -> List[float]:
    """
    Forecasts epidemic progression based on case history.
    
    Args:
        current_cases: Current number of cases
        case_history: Historical case data
        days_to_forecast: Number of days to forecast into the future
        
    Returns:
        List[float]: Forecasted case numbers for the specified number of days
    """
    # Simplified linear regression approach if enough history
    history = [h for h in case_history if h is not None] # Clean Nones
    if len(history) >= 5:
        try:
            X = np.arange(len(history)).reshape(-1, 1)
            y = np.array(history)
            model = LinearRegression()
            model.fit(X, y)
            # Forecast future days
            future_X = np.arange(len(history), len(history) + days_to_forecast).reshape(-1, 1)
            forecast = model.predict(future_X)
            # Ensure non-negative forecast
            forecast = np.maximum(0, forecast)
            return forecast.tolist()
        except Exception:
            # Fallback if regression fails
            pass # Fall through to simpler method

    # Simpler trend projection if not enough data or regression failed
    if len(history) >= 2:
        # Use average change over last few points
        recent_history = history[-min(5, len(history)):]
        if len(recent_history) >= 2:
            avg_change = (recent_history[-1] - recent_history[0]) / (len(recent_history) -1)
        else:
            avg_change = 0 # Default if only one point available after filtering
    elif len(history) == 1:
        avg_change = 0 # No trend possible
    else: # No history
        avg_change = 0
        history = [current_cases] # Start history with current

    # Project forward, ensuring non-negative
    forecast = []
    last_val = history[-1] if history else current_cases # Use current if history is empty
    for _ in range(days_to_forecast):
        next_val = last_val + avg_change
        forecast.append(max(0, next_val))
        last_val = max(0, next_val) # Use the non-negative value for next step
    return forecast


def disruption_prediction_tool(
    historical_disruptions: List[Dict], current_day: int, look_ahead_days: int = 14
) -> Dict:
    """
    Predicts likelihood of disruptions based on historical patterns.
    
    Args:
        historical_disruptions: List of disruption events with type, timing, and target information
        current_day: Current simulation day
        look_ahead_days: Number of days to forecast disruptions for
        
    Returns:
        Dict: Predicted disruption probabilities by type and target
    """
    predictions = {"manufacturing": {}, "transportation": {}}
    manu_counts = {}
    trans_counts = {}
    # Count unique disruption *periods* rather than just start days
    manu_disruption_periods = set() # Store tuples of (type, target_id, start_day)
    trans_disruption_periods = set()

    for d in historical_disruptions:
        key = (d["type"], d.get("drug_id", d.get("region_id")), d["start_day"])
        if d["type"] == "manufacturing" and "drug_id" in d:
            if key not in manu_disruption_periods:
                manu_counts[d["drug_id"]] = manu_counts.get(d["drug_id"], 0) + 1
                manu_disruption_periods.add(key)
        elif d["type"] == "transportation" and "region_id" in d:
            if key not in trans_disruption_periods:
                trans_counts[d["region_id"]] = trans_counts.get(d["region_id"], 0) + 1
                trans_disruption_periods.add(key)

    days_passed = max(1, current_day) # Avoid division by zero if current_day is 0
    # Calculate probability using Poisson assumption: P(at least one) = 1 - P(zero)
    # Rate (lambda) = average number of events per day
    for drug_id, count in manu_counts.items():
        rate = count / days_passed
        prob_zero_in_lookahead = math.exp(-rate * look_ahead_days)
        prob_at_least_one = 1 - prob_zero_in_lookahead
        predictions["manufacturing"][str(drug_id)] = min(0.95, max(0, prob_at_least_one)) # Ensure 0 <= prob <= 0.95

    for region_id, count in trans_counts.items():
        rate = count / days_passed
        prob_zero_in_lookahead = math.exp(-rate * look_ahead_days)
        prob_at_least_one = 1 - prob_zero_in_lookahead
        predictions["transportation"][str(region_id)] = min(0.95, max(0, prob_at_least_one)) # Ensure 0 <= prob <= 0.95

    return predictions