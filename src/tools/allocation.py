# --- START OF FILE src/tools/allocation.py ---

"""
Allocation and order quantity optimization tools for pandemic supply chain simulation.
"""

import numpy as np
from typing import Dict, List

# Note: allocation_priority_tool remains largely unchanged as per user prompt instructions,
# focusing changes on optimal_order_quantity_tool.
# The manufacturer agent's fallback logic now uses this tool differently (Step 3).
def allocation_priority_tool(
    drug_info: Dict, # Information about the specific drug being allocated
    region_requests: Dict[int, float], # {region_id: requested_amount}
    region_cases: Dict[int, float], # {region_id: current_cases} - Used for prioritization
    available_inventory: float
) -> Dict[int, float]:
    """
    Determines allocation of limited inventory across regions based on
    requests, drug criticality, and regional case loads. (Prioritizes higher requests, higher cases, higher criticality).

    Args:
        drug_info: Dict containing info like 'criticality_value'.
        region_requests: Requested amounts by region ID.
        region_cases: Current case loads by region ID.
        available_inventory: Total inventory available for allocation.

    Returns:
        Dict[int, float]: Calculated allocation amounts by region ID. Keys match input region_requests.
    """
    # Ensure inputs are valid
    try:
        available_inventory = max(0.0, float(available_inventory))
    except (ValueError, TypeError):
        available_inventory = 0.0

    # Validate and clean requests and cases, ensuring keys are integers
    valid_requests = {}
    clean_region_cases = {}
    original_region_ids = set(region_requests.keys()) # Keep track of all originally requested regions

    for r_id, req in region_requests.items():
        try:
            r_id_int = int(r_id)
            valid_requests[r_id_int] = max(0.0, float(req)) if req is not None else 0.0
        except (ValueError, TypeError):
            valid_requests[r_id_int] = 0.0 # Treat invalid requests as 0

    for r_id, cases in region_cases.items():
         try:
             r_id_int = int(r_id)
             clean_region_cases[r_id_int] = max(0.0, float(cases)) if cases is not None else 0.0
         except (ValueError, TypeError):
             clean_region_cases[r_id_int] = 0.0 # Treat invalid case counts as 0


    total_requested = sum(valid_requests.values())

    # Initialize result dict with all original region IDs mapped to 0.0
    final_output = {r_id: 0.0 for r_id in original_region_ids}

    # If nothing valid requested or no inventory, return zero allocations for all original regions
    if total_requested <= 1e-6 or available_inventory <= 1e-6:
        return final_output

    # If enough inventory, fulfill all valid (non-negative) requests
    if total_requested <= available_inventory:
        # Update final_output only for the valid requests
        for r_id, req in valid_requests.items():
            final_output[r_id] = req
        return final_output

    # Not enough inventory - Calculate priorities for proportional allocation
    allocations = {} # Temporary dict for calculated allocations
    priorities = {}
    total_priority = 0.0
    drug_criticality = drug_info.get("criticality_value", 1) # Default criticality = 1

    for region_id, request in valid_requests.items():
        if request <= 1e-6: continue # Skip zero requests in priority calculation

        # Weighting factors (can be tuned)
        # Use cleaned case data, ensure log1p handles 0 cases correctly
        regional_cases = clean_region_cases.get(region_id, 0.0)
        case_weight = 1.0 + np.log1p(regional_cases) # Log scale for cases, +1 smooths low vals
        criticality_weight = drug_criticality**2 # Square criticality to give more weight

        # Priority = Request size * Case Load Weight * Criticality Weight
        priority = request * case_weight * criticality_weight
        priorities[region_id] = max(0.0, priority) # Ensure non-negative
        total_priority += priorities[region_id]

    # Allocate proportionally based on priority
    if total_priority > 1e-6:
        for region_id, priority in priorities.items():
            proportion = priority / total_priority
            # Allocate proportionally, but cap at the original request
            allocated_amount = min(valid_requests[region_id], proportion * available_inventory)
            allocations[region_id] = allocated_amount
    else:
        # Fallback: If total priority is zero (e.g., zero cases everywhere, zero requests), allocate equally among positive requesters
        positive_requesters = {r_id: req for r_id, req in valid_requests.items() if req > 1e-6}
        num_positive_requesters = len(positive_requesters)
        if num_positive_requesters > 0:
            equal_share = available_inventory / num_positive_requesters
            for region_id, request_amount in positive_requesters.items():
                allocations[region_id] = min(request_amount, equal_share)
        # else: allocations remains empty, final_output remains all zeros

    # Final check to ensure total allocated doesn't exceed available due to potential float issues
    total_allocated = sum(allocations.values())
    if total_allocated > available_inventory * 1.0001: # Allow tiny tolerance
        if total_allocated > 1e-6: # Avoid division by zero
            scale_down = available_inventory / total_allocated
            allocations = {r: a * scale_down for r, a in allocations.items()}
        else:
            allocations = {r: 0.0 for r in allocations} # Should not happen


    # Update the final_output dict with calculated allocations, ensuring non-negative
    for r_id, alloc in allocations.items():
        if r_id in final_output: # Ensure the region was part of the original request
            final_output[r_id] = max(0.0, alloc)

    return final_output


# --- STEP 2 CHANGE: Enhanced Tool ---
def optimal_order_quantity_tool(
    inventory_level: float,
    pipeline_quantity: float,
    daily_demand_forecast: List[float], # Expects a list of forecasted demands
    lead_time: int = 3,
    safety_stock_factor: float = 1.5 # Base factor (can be influenced by criticality upstream)
) -> float:
    """
    Calculates optimal order quantity using an order-up-to inventory policy,
    incorporating safety stock adjusted for DEMAND FORECAST TREND.

    Args:
        inventory_level: Current on-hand inventory.
        pipeline_quantity: Inventory already ordered but not yet received.
        daily_demand_forecast: Forecasted daily demand for future periods (list of floats).
        lead_time: Expected lead time for orders (days).
        safety_stock_factor: Base factor to apply to demand variability for safety stock.

    Returns:
        float: Optimal order quantity.
    """
    # Input Validation and Cleaning
    try: inventory_level = max(0.0, float(inventory_level))
    except (ValueError, TypeError): inventory_level = 0.0
    try: pipeline_quantity = max(0.0, float(pipeline_quantity))
    except (ValueError, TypeError): pipeline_quantity = 0.0
    try: lead_time = max(1, int(lead_time))
    except (ValueError, TypeError): lead_time = 1
    try: safety_stock_factor = max(1.0, float(safety_stock_factor)) # Ensure factor is at least 1
    except (ValueError, TypeError): safety_stock_factor = 1.0

    valid_forecast = []
    if daily_demand_forecast: # Check if list is not None or empty
        for d in daily_demand_forecast:
             try: valid_forecast.append(max(0.0, float(d)))
             except (ValueError, TypeError): continue # Skip non-numeric forecasts


    # Define planning horizon: Lead Time + Review Period (assuming daily review = 1 day)
    review_period = 1
    planning_horizon = lead_time + review_period

    # --- Calculate Trend Multiplier based on Forecast ---
    trend_multiplier = 1.0
    sensitivity_factor = 0.75 # TUNABLE: How strongly trend impacts safety stock
    max_trend_multiplier = 3.5 # TUNABLE: Maximum boost from trend

    if len(valid_forecast) >= planning_horizon and planning_horizon >= 2:
        # More robust trend calculation: slope of linear regression over the planning horizon
        try:
            X = np.arange(planning_horizon).reshape(-1, 1)
            y = np.array(valid_forecast[:planning_horizon])
            # Handle cases with zero variance (e.g., constant forecast)
            if np.std(y) > 1e-9:
                model = np.polyfit(X.flatten(), y, 1) # Linear fit (slope is model[0])
                slope = model[0]
                avg_demand_in_period = np.mean(y) if len(y)>0 else 0.0
                if avg_demand_in_period > 1e-6: # Avoid division by zero
                    relative_slope = slope / avg_demand_in_period
                    # Scale multiplier by magnitude of relative slope and period length
                    trend_multiplier = 1.0 + max(0, relative_slope) * sensitivity_factor * planning_horizon
                    trend_multiplier = min(trend_multiplier, max_trend_multiplier)
            # else: keep trend_multiplier = 1.0 if demand is constant/zero
        except Exception: # Fallback if regression fails
            trend_multiplier = 1.0
    elif len(valid_forecast) >= 2: # Simpler trend if forecast too short
         avg_change = (valid_forecast[-1] - valid_forecast[0]) / max(1, len(valid_forecast) - 1)
         avg_demand = np.mean(valid_forecast) if len(valid_forecast)>0 else 0.0
         if avg_demand > 1e-6:
              relative_change = avg_change / avg_demand
              trend_multiplier = 1.0 + max(0, relative_change) * sensitivity_factor * planning_horizon
              trend_multiplier = min(trend_multiplier, max_trend_multiplier)

    # --- Calculate Demand Metrics over Planning Horizon ---
    demand_during_horizon = 0.0
    std_dev_demand_horizon = 0.0

    if not valid_forecast:
        # Handle case with no valid forecast data
        demand_during_horizon = 0.0
        std_dev_demand_horizon = 0.0 # No variability known
    elif len(valid_forecast) < planning_horizon:
        # Extrapolate if forecast is too short
        avg_daily_demand = np.mean(valid_forecast)
        known_demand = sum(valid_forecast)
        extrapolated_demand = avg_daily_demand * max(0, planning_horizon - len(valid_forecast))
        demand_during_horizon = known_demand + extrapolated_demand

        # Estimate standard deviation based on available history
        std_dev_daily = np.std(valid_forecast) if len(valid_forecast) > 1 else (valid_forecast[0] * 0.3) # Assume 30% volatility if only one point
        std_dev_demand_horizon = std_dev_daily * np.sqrt(planning_horizon)
    else:
        # Use the forecast directly for the planning horizon
        demand_during_horizon = sum(valid_forecast[:planning_horizon])
        std_dev_daily = np.std(valid_forecast[:planning_horizon])
        std_dev_demand_horizon = std_dev_daily * np.sqrt(planning_horizon)

    # Ensure demand and std dev are non-negative
    demand_during_horizon = max(0.0, demand_during_horizon)
    std_dev_demand_horizon = max(0.0, std_dev_demand_horizon)

    # --- Calculate Safety Stock ---
    # Base safety stock accounts for variability
    base_safety_stock = safety_stock_factor * std_dev_demand_horizon
    # Adjusted safety stock accounts for trend
    adjusted_safety_stock = base_safety_stock * trend_multiplier

    # --- Calculate Order-Up-To Level ---
    target_level = demand_during_horizon + adjusted_safety_stock

    # --- Calculate Inventory Position ---
    inventory_position = inventory_level + pipeline_quantity

    # --- Calculate Order Quantity ---
    order_quantity = target_level - inventory_position

    # Ensure non-negative order quantity
    return max(0.0, order_quantity)

# --- END OF FILE src/tools/allocation.py ---