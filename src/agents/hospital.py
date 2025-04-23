# --- START OF FILE src/agents/hospital.py ---

"""
Hospital agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools

class HospitalAgent(OpenAIPandemicLLMAgent):
    """LLM-powered hospital agent using OpenAI."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools,
        openai_integration,
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add interface
        ):
        super().__init__(
            "hospital",
            region_id,
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
            )
        # Hospitals typically don't need num_regions, but could be added if needed

    def decide(self, observation: Dict) -> Dict:
        """Make ordering decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools ---
        # Hospital doesn't need blockchain cases directly for its order decision logic (uses projected demand)
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Demand-based
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to distributor) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {"hospital_orders": {self.agent_id: order_decisions}}

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from distributor using OpenAI, with enhanced fallback and rules."""
        decision_type = "order"
        # Prompt uses cleaned observation (no direct cases/trend)
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           order_amount = max(0.0, float(amount))
                           processed_llm[drug_id] = order_amount
                      else:
                          if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {self.agent_type} {decision_type} item '{drug_id_key}': {amount} -> {e} (Region {self.agent_id}). Skipping.[/]")

             if processed_llm:
                llm_success = True
                order_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based assessment & optimal order.[/]")
             # Fallback: Use criticality assessment and optimal order tool (based on projected demand)
             for drug_id in range(num_drugs):
                  inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                  pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  criticality = drug_info.get("criticality_value", 1)

                  # Use hospital's own projected demand from observation
                  # Projected demand IS still present in the observation dict
                  next_day_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                  next_day_projected_demand = max(0, float(next_day_projected_demand))

                  # Estimate lead time based on disruption risk
                  transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                  base_lead_time = 1 # Dist -> Hosp is short
                  lead_time = base_lead_time + int(round(transport_risk * 3))
                  lead_time = max(1, lead_time) # Ensure lead time is at least 1

                  # Create forecast list based on own projected demand.
                  # The tool will use the trend within this list implicitly.
                  demand_forecast_for_tool = [next_day_projected_demand] * (lead_time + 1) # +1 review period

                  # Ensure forecast has at least one element if it's empty
                  if not demand_forecast_for_tool:
                       demand_forecast_for_tool = [next_day_projected_demand]

                  # Call the optimal order tool
                  order_qty = self._run_optimal_order_quantity_tool(
                      inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                  )
                  order_decisions[drug_id] = order_qty


        # --- Apply Rule-Based Adjustments (No changes needed, assessment based on local history) ---
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        for drug_id in range(num_drugs): # Ensure we assess all drugs
             # Run assessment tool regardless of LLM success/failure
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             # Get histories (ensure they are lists)
             stockout_hist = observation.get("stockout_history", [])
             demand_hist = observation.get("demand_history", [])
             stockout_hist = stockout_hist if isinstance(stockout_hist, list) else []
             demand_hist = demand_hist if isinstance(demand_hist, list) else []

             unfulfilled = sum(s.get('unfulfilled', 0) for s in stockout_hist if isinstance(s, dict))
             total_demand_hist = sum(d.get('demand', 0) for d in demand_hist if isinstance(d, dict))

             # Call assessment tool (verbose printing inside base class method)
             situation = self._run_criticality_assessment_tool(
                 drug_info, stockout_hist, unfulfilled, max(1, total_demand_hist) # Avoid division by zero
             )

             # --- Apply Emergency Override / Multiplier Logic ---
             crit_category = situation.get("category", "")
             base_multiplier = 1.0 # Default

             # Determine base multiplier based on category assessment
             if crit_category == "Critical Emergency": base_multiplier = 3.0
             elif crit_category == "Severe Shortage": base_multiplier = 2.0
             elif crit_category == "Moderate Concern": base_multiplier = 1.5

             # Determine emergency boost target multiplier (overrides base if higher)
             emergency_boost = 1.0 # Default if not emergency/severe
             if crit_category == "Critical Emergency":
                 emergency_boost = 2.5
             elif crit_category == "Severe Shortage":
                 emergency_boost = 1.8

             # Use the *highest* relevant multiplier
             final_multiplier = max(base_multiplier, emergency_boost)

             if abs(final_multiplier - 1.0) > 0.01: # Apply if multiplier > 1
                 if self.verbose:
                     reason = "EMERGENCY override boost" if final_multiplier == emergency_boost and emergency_boost > base_multiplier else "criticality assessment multiplier"
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying {reason} (Category: {crit_category}, Factor: {final_multiplier:.2f}).[/]")
                     rules_applied_flag = True
                 current_order = order_decisions.get(drug_id, 0) # Get current value or 0 if missing
                 order_decisions[drug_id] = current_order * final_multiplier


        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             # Format for printing
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             # Only print comparison if rules actually ran and changed something significantly
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else: # Print final decision even if rules didn't change it, but rules were checked
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose: # Print the decision even if no rules applied / checked (e.g., LLM succeeded first)
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        # Return only positive, integer-keyed orders
        return {int(k): v for k, v in order_decisions.items() if v > 0.01} # Use threshold to avoid tiny orders


def create_openai_hospital_agent(
    region_id,
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added interface
):
    """Create a hospital agent powered by OpenAI."""
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