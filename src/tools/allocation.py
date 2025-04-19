"""
Allocation and order quantity optimization tools for pandemic supply chain simulation.
"""

import numpy as np
from typing import Dict, List

def allocation_priority_tool(
    drug_info: Dict, # Should be Dict for a single drug
    region_requests: Dict[int, float],
    region_cases: Dict[int, float],
    available_inventory: float
) -> Dict[int, float]:
    """
    Determines optimal allocation of limited inventory across regions based on
    drug criticality, region requests, and case loads.
    
    Args:
        drug_info: Information about the drug being allocated
        region_requests: Requested amounts by region
        region_cases: Current case loads by region
        available_inventory: Total inventory available for allocation
        
    Returns:
        Dict[int, float]: Optimal allocation amounts by region
    """
    # If nothing requested or no inventory, return zero allocations
    total_requested = sum(region_requests.values())
    if total_requested <= 0: 
        return {r: 0 for r in region_requests}

    # If enough inventory, fulfill all valid requests
    if total_requested <= available_inventory:
        # Filter out non-positive requests before returning
        return {r: max(0, a) for r, a in region_requests.items()}

    # Basic proportional allocation based on request size, cases, and criticality
    allocations = {}
    
    # Simple weighting based on cases and criticality
    drug_criticality = drug_info.get("criticality_value", 1)
    priorities = {}
    total_priority = 0
    
    for region_id, request in region_requests.items():
        case_weight = region_cases.get(region_id, 0) + 1  # Add 1 to avoid zero weight
        priority = request * case_weight * drug_criticality
        priorities[region_id] = priority
        total_priority += priority
        
    # Allocate proportionally based on priority
    for region_id, priority in priorities.items():
        proportion = priority / total_priority if total_priority > 0 else 0
        allocations[region_id] = min(region_requests[region_id], proportion * available_inventory)
        
    return allocations


def optimal_order_quantity_tool(
    inventory_level: float,
    pipeline_quantity: float,
    daily_demand_forecast: List[float],
    lead_time: int = 3,
    safety_stock_factor: float = 1.5 # Base factor
) -> float:
    """
    Calculates optimal order quantity using a basic order-up-to inventory policy
    with safety stock based on demand variability.
    
    Args:
        inventory_level: Current on-hand inventory
        pipeline_quantity: Inventory already ordered but not yet received
        daily_demand_forecast: Forecasted daily demand for future periods
        lead_time: Expected lead time for orders (days)
        safety_stock_factor: Factor to apply to demand variability for safety stock
        
    Returns:
        float: Optimal order quantity
    """
    # Ensure inputs are non-negative floats
    inventory_level = max(0.0, float(inventory_level))
    pipeline_quantity = max(0.0, float(pipeline_quantity))
    lead_time = max(1, int(lead_time)) # Lead time should be at least 1 day

    # Ensure forecast contains non-negative numbers
    valid_forecast = [max(0.0, float(d)) for d in daily_demand_forecast if d is not None]

    # Calculate demand during lead time + 1 review period (assuming daily review)
    review_period = 1
    total_period = lead_time + review_period
    if not valid_forecast:
        demand_during_period = 0
        std_dev_demand = 0
    elif len(valid_forecast) < total_period:
        # If forecast is shorter than needed period, extrapolate or use average
        avg_daily_demand = np.mean(valid_forecast) if valid_forecast else 0
        demand_during_period = sum(valid_forecast) + avg_daily_demand * max(0, total_period - len(valid_forecast))
        std_dev_daily = np.std(valid_forecast) if len(valid_forecast) > 1 else (valid_forecast[0] * 0.2 if valid_forecast else 0)
        std_dev_demand = std_dev_daily * np.sqrt(total_period)
    else:
        # Demand during lead time + review period
        demand_during_period = sum(valid_forecast[:total_period])
        # Standard deviation of demand *over the lead time + review period*
        # Simple approx: sqrt(period) * std_dev_daily
        std_dev_daily = np.std(valid_forecast[:total_period])
        std_dev_demand = std_dev_daily * np.sqrt(total_period)

    # Calculate safety stock using the factor (representing service level target)
    # Factor applied to the standard deviation of demand over the lead time + review period
    safety_stock = safety_stock_factor * std_dev_demand

    # Calculate target inventory (order-up-to level)
    target_level = demand_during_period + safety_stock

    # Calculate current inventory position
    inventory_position = inventory_level + pipeline_quantity

    # Calculate order quantity needed to reach target level
    order_quantity = target_level - inventory_position

    # Return non-negative order quantity
    return max(0.0, order_quantity)