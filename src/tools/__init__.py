"""
Tools for the pandemic supply chain simulation agents.

This package provides various tools for forecasting, allocation, and situation assessment
that can be used by supply chain agents to make informed decisions.
"""

from src.tools.forecasting import epidemic_forecast_tool, disruption_prediction_tool
from src.tools.allocation import allocation_priority_tool, optimal_order_quantity_tool
from src.tools.assessment import criticality_assessment_tool

# For backwards compatibility, provide the PandemicSupplyChainTools class
class PandemicSupplyChainTools:
    """
    Collection of decision-making tools for supply chain agents.
    This class provides a unified interface to the various tool functions.
    """
    
    @staticmethod
    def epidemic_forecast_tool(current_cases, case_history, days_to_forecast=14):
        """Forecasts epidemic progression."""
        return epidemic_forecast_tool(current_cases, case_history, days_to_forecast)
    
    @staticmethod
    def disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days=14):
        """Predicts likelihood of disruptions."""
        return disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days)
    
    @staticmethod
    def allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory):
        """Determines optimal allocation."""
        return allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory)
    
    @staticmethod
    def optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time=3, safety_stock_factor=1.5):
        """Calculates optimal order quantity."""
        return optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time, safety_stock_factor)
    
    @staticmethod
    def criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand):
        """Assesses situation criticality."""
        return criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand)

# Export the functions directly for direct import
__all__ = [
    'PandemicSupplyChainTools',
    'epidemic_forecast_tool',
    'disruption_prediction_tool',
    'allocation_priority_tool',
    'optimal_order_quantity_tool',
    'criticality_assessment_tool'
]