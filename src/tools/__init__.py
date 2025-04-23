# --- START OF FILE src/tools/__init__.py ---

"""
Tools for the pandemic supply chain simulation agents.

This package provides various tools for forecasting, allocation, and situation assessment
that can be used by supply chain agents to make informed decisions.
"""

from typing import Dict, List, Optional

# Import core tool functions from other modules in the package
from src.tools.forecasting import epidemic_forecast_tool, disruption_prediction_tool
from src.tools.allocation import allocation_priority_tool, optimal_order_quantity_tool
from src.tools.assessment import criticality_assessment_tool

# Import BlockchainInterface for type hinting if needed, handle import error
# This allows the tool function signature to be correct even if blockchain dependencies are not installed.
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Define as None if import fails

# Optional: Import console for potential logging within tools if desired
# from config import console


# Define the wrapper class for compatibility and unified access
class PandemicSupplyChainTools:
    """
    Collection of decision-making tools for supply chain agents.
    This class provides a unified interface to the various tool functions
    via static methods.
    """

    @staticmethod
    def epidemic_forecast_tool(current_cases, case_history, days_to_forecast=14):
        """Forecasts epidemic progression."""
        # Note: This tool's usage might change based on Strategy 1 implementation
        # where agents don't get direct case counts. The agent calling this
        # might need to provide projected demand or other context instead.
        # The function itself remains as originally defined for now.
        return epidemic_forecast_tool(current_cases, case_history, days_to_forecast)

    @staticmethod
    def disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days=14):
        """Predicts likelihood of disruptions based on historical events."""
        return disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days)

    @staticmethod
    def allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory):
        """
        Determines allocation based on requests, drug criticality, and regional case loads.
        """
        # This tool is used by the Manufacturer agent's fallback logic and
        # potentially by the local simulation fallback if blockchain isn't used.
        return allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory)

    @staticmethod
    def optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time=3, safety_stock_factor=1.5):
        """
        Calculates optimal order quantity using an order-up-to policy,
        incorporating safety stock adjusted for forecast trend.
        """
        return optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time, safety_stock_factor)

    @staticmethod
    def criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand):
        """
        Assesses the criticality of a drug supply situation based on stockout history
        and unfulfilled demand.
        """
        return criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand)

    # --- NEW TOOL METHOD (Strategy 1) ---
    @staticmethod
    def get_blockchain_regional_cases_tool(
        blockchain_interface: Optional[BlockchainInterface],
        num_regions: int
    ) -> Optional[Dict[int, int]]:
        """
        Queries the blockchain for the latest case counts for all regions.

        This tool fetches data considered a "trusted source of truth" from the
        blockchain, intended to be used by agents instead of relying solely on
        simulation-internal data passed via observations.

        Args:
            blockchain_interface: The initialized BlockchainInterface object.
                                  If None, the tool cannot operate and returns None.
            num_regions: The total number of regions in the simulation, needed to
                         iterate through and query each region's data.

        Returns:
            A dictionary mapping {region_id: case_count} if successful.
            Returns None if the blockchain interface is not available or if
            querying fails critically (though it attempts to return partial
            data with defaults for minor failures).
            Returns an empty dict if num_regions is 0.
        """
        if blockchain_interface is None:
            # Logging might be better handled by the calling agent
            # print("[TOOL WARNING] Blockchain tool called, but interface is None.")
            return None
        if num_regions <= 0:
            return {} # No regions to query

        regional_cases = {}
        all_successful = True
        for region_id in range(num_regions):
            # The get_regional_case_count method in BlockchainInterface should handle retries/errors
            case_count = blockchain_interface.get_regional_case_count(region_id)
            if case_count is None:
                # Interface failed to get data for this region after retries
                all_successful = False
                regional_cases[region_id] = 0 # Provide a default value (0 cases) for robustness
                # Logging handled within interface or calling agent
            else:
                regional_cases[region_id] = case_count

        if not all_successful:
            # Log that some regions failed, but return the dictionary with defaults
            # print(f"[TOOL WARNING] Blockchain tool: Failed to retrieve case counts for some regions. Using 0 default for failed regions.")
            pass # Let calling agent decide how to handle partial success

        return regional_cases


# Export the class and the individual functions for flexibility
__all__ = [
    'PandemicSupplyChainTools',          # The wrapper class
    'epidemic_forecast_tool',            # Individual functions
    'disruption_prediction_tool',
    'allocation_priority_tool',
    'optimal_order_quantity_tool',
    'criticality_assessment_tool',
    'get_blockchain_regional_cases_tool' # The new blockchain query tool function
]
# --- END OF FILE src/tools/__init__.py ---