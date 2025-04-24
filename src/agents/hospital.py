# --- START OF FILE src/agents/hospital.py ---

"""
Hospital agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
import json # For checking rule changes
import numpy as np # Used in fallback logic

# Import base agent class
from .base import OpenAIPandemicLLMAgent
# Import configuration and tools
from config import Colors
from src.tools import PandemicSupplyChainTools # Import the tools class

# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Define as None if import fails


class HospitalAgent(OpenAIPandemicLLMAgent):
    """LLM-powered hospital agent using OpenAI."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools, # Expect tools instance
        openai_integration,
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add interface
        ):
        """
        Initializes the Hospital Agent.

        Args:
            region_id (int): The region this hospital operates in.
            tools (PandemicSupplyChainTools): Instance of the tools collection.
            openai_integration: Instance of the OpenAI integration class.
            memory_length (int): How many past observations to remember.
            verbose (bool): Whether to print detailed logs.
            console: Rich console instance for printing.
            blockchain_interface (Optional[BlockchainInterface]): Blockchain interface instance.
        """
        super().__init__(
            agent_type="hospital",
            agent_id=region_id, # region_id serves as the agent_id for hospitals
            tools=tools,
            openai_integration=openai_integration,
            memory_length=memory_length,
            verbose=verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
            )
        # Hospitals typically don't need num_regions directly for their logic,
        # but the base class might infer it if needed for tools.

    def decide(self, observation: Dict) -> Dict:
        """
        Makes ordering decisions for the hospital based on the current observation.

        Args:
            observation (Dict): The current state observation for this hospital.

        Returns:
            Dict: A dictionary containing the orders placed by this hospital, structured as:
                  {"hospital_orders": {self.agent_id: {drug_id: amount}}}
        """
        self.add_to_memory(observation) # Store current state (includes inferring num_regions if needed)

        agent_name = self._get_agent_name()
        agent_color = self._get_agent_color()
        self._print(f"\n[{agent_color}]🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")

        # --- Use Tools ---
        # Hospitals rely primarily on their projected demand and inventory status.
        # They don't typically query system-wide blockchain cases directly for their ordering logic.
        # Forecast tool now uses projected demand from observation.
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Uses proj. demand
        disruption_predictions = self._run_disruption_prediction_tool(observation) # Predicts transport risk

        # --- Ordering decisions (to distributor) ---
        # Passes tool outputs for context; fallback relies heavily on observation data and optimal order tool.
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly: { "hospital_orders": { hospital_agent_id: { drug_id: amount } } }
        # Hospital agent_id is its region_id
        return {"hospital_orders": {self.agent_id: order_decisions}}

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict[int, float]:
        """
        Determine order quantities from the distributor using OpenAI, with fallback logic and rule-based adjustments.

        Args:
            observation (Dict): Current state observation.
            epidemic_forecast_tool_output (List[float]): Output from the forecast tool (currently based on projected demand).
            disruption_predictions (Dict): Output from the disruption prediction tool.

        Returns:
            Dict[int, float]: Dictionary mapping drug_id (int) to order quantity (float).
        """
        decision_type = "order"
        agent_name = self._get_agent_name() # For logging

        # Create prompt for LLM (observation cleaned in base class method)
        prompt = self._create_decision_prompt(observation, decision_type)

        # Call LLM API
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

        order_decisions = {} # Initialize {drug_id: amount}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        # --- Process LLM Decision ---
        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           order_amount = max(0.0, float(amount)) # Ensure non-negative
                           processed_llm[drug_id] = order_amount
                      else:
                          if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {agent_name} {decision_type}.[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {agent_name} {decision_type} item '{drug_id_key}': {amount} -> {e}. Skipping.[/]")

             if processed_llm: # If any valid decisions were processed
                llm_success = True
                order_decisions = processed_llm # Use LLM decisions as the starting point

        # --- Fallback Logic ---
        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {agent_name} {decision_type} decision failed/invalid. Using fallback: Rule-based assessment & optimal order.[/]")
             # Fallback: Use rule-based criticality assessment and optimal order tool (based on projected demand)
             for drug_id in range(num_drugs):
                  inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                  pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  criticality = drug_info.get("criticality_value", 1) # Default criticality if missing

                  # Use hospital's own projected demand from observation for the next day
                  next_day_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                  next_day_projected_demand = max(0, float(next_day_projected_demand))

                  # Estimate lead time based on disruption risk (Distributor -> Hospital)
                  transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0) # Use agent's region_id
                  base_lead_time = 1 # Assume short base lead time for Dist -> Hosp
                  # Adjust lead time based on risk (more sensitive scaling can be tuned)
                  lead_time = base_lead_time + int(round(transport_risk * 3)) # e.g., risk 0.5 adds ~1.5 days
                  lead_time = max(1, lead_time) # Ensure lead time is at least 1 day

                  # Create a simple demand forecast list for the tool based on the *next day's* projected demand
                  # Tool's enhanced logic will handle trend implicitly within this forecast list.
                  planning_horizon = lead_time + 1 # Lead Time + Review Period (1)
                  demand_forecast_for_tool = [next_day_projected_demand] * planning_horizon

                  # Call the optimal order tool (verbose printing happens inside the base class method)
                  order_qty = self._run_optimal_order_quantity_tool(
                      inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                  )
                  order_decisions[drug_id] = order_qty # Add fallback decision

        # --- Rule-Based Adjustments (Applied AFTER LLM/Fallback) ---
        decisions_before_rules = order_decisions.copy() # Store decision state before rules
        rules_applied_flag = False

        for drug_id in list(order_decisions.keys()): # Iterate safely over keys
             # Get necessary info from observation, providing defaults
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             stockout_hist = observation.get("stockout_history", []) or [] # Ensure list
             demand_hist = observation.get("demand_history", []) or []    # Ensure list

             # Calculate metrics for the assessment tool safely
             unfulfilled = sum(s.get('unfulfilled', 0) for s in stockout_hist if isinstance(s, dict))
             total_demand_hist = sum(d.get('demand', 0) for d in demand_hist if isinstance(d, dict))

             # Call assessment tool (verbose printing inside base class method)
             situation = self._run_criticality_assessment_tool(
                 drug_info, stockout_hist, unfulfilled, max(1.0, total_demand_hist) # Avoid division by zero
             )

             # --- Apply Tuned-Down Emergency Override / Multiplier Logic ---
             crit_category = situation.get("category", "Normal Operations") # Default category
             final_multiplier = 1.0 # Default multiplier

             # Determine multiplier based on assessment category (less aggressive)
             if crit_category == "Critical Emergency": final_multiplier = 2.0 # WAS 3.0
             elif crit_category == "Severe Shortage": final_multiplier = 1.5 # WAS 2.0
             elif crit_category == "Moderate Concern": final_multiplier = 1.2 # WAS 1.5
             # No explicit multiplier for "Potential Issue" or "Normal Operations"

             # Apply the multiplier if it's greater than 1
             if abs(final_multiplier - 1.0) > 0.01:
                 if self.verbose:
                     reason = "criticality assessment multiplier"
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying {reason} (Category: {crit_category}, Factor: {final_multiplier:.2f}).[/]")
                     rules_applied_flag = True

                 current_order = order_decisions.get(drug_id, 0.0) # Get current order or default to 0.0
                 # Ensure the key exists before multiplication, although list(keys()) makes this safer
                 order_decisions[drug_id] = current_order * final_multiplier

        # --- Final Logging ---
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()} # Format for printing
             log_prefix = ""
             changed = rules_applied_flag and json.dumps({k: f"{v:.1f}" for k, v in decisions_before_rules.items()}) != json.dumps(print_after)

             if not llm_success:
                 log_prefix = f"[{Colors.FALLBACK}][FALLBACK FINAL Decision]"
             elif rules_applied_flag and changed:
                 log_prefix = f"[{Colors.RULE}][RULE FINAL]"
             elif rules_applied_flag: # Rules ran but didn't change output significantly
                 log_prefix = f"[{Colors.DECISION}][FINAL Decision]"
             else: # LLM succeeded, no rules applied/changed
                 log_prefix = f"[{Colors.DECISION}][FINAL Decision]"

             # Print detailed comparison only if rules ran and changed the outcome
             if rules_applied_flag and changed:
                 print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
                 self._print(f"{log_prefix} {agent_name} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                 rule_status = "(Rules checked, no change)" if rules_applied_flag else ""
                 self._print(f"{log_prefix} {agent_name} - {decision_type} {rule_status}:[/] {print_after}")

        # Return only positive, integer-keyed orders, filtering negligible amounts
        return {int(k): v for k, v in order_decisions.items() if v > 0.01}


# --- Factory Function ---

def create_openai_hospital_agent(
    region_id: int,
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    memory_length: int = 10,
    verbose: bool = True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added interface
) -> HospitalAgent:
    """
    Factory function to create a hospital agent powered by OpenAI.

    Args:
        region_id (int): The region ID for the hospital.
        tools (PandemicSupplyChainTools): The shared tools instance.
        openai_integration: The OpenAI integration instance.
        memory_length (int): Length of the agent's memory.
        verbose (bool): Verbosity flag.
        console: Rich console for logging.
        blockchain_interface (Optional[BlockchainInterface]): Blockchain interface instance.

    Returns:
        HospitalAgent: An initialized HospitalAgent instance.
    """
    return HospitalAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
        )

# --- END OF FILE src/agents/hospital.py ---