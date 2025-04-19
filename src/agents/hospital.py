# --- START OF FILE src/agents/hospital.py ---

"""
Hospital agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json 

class HospitalAgent(OpenAIPandemicLLMAgent):
    """LLM-powered hospital agent using OpenAI."""

    def __init__(self, region_id, tools, openai_integration, memory_length=10, verbose=True, console=None):
        super().__init__("hospital", region_id, tools, openai_integration, memory_length, verbose, console=console)

    def decide(self, observation: Dict) -> Dict:
        """Make ordering decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        # Tools are run inside the decision methods if needed for fallback or rules
        epidemic_forecast = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to distributor) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast, disruption_predictions)

        # Structure the output correctly
        return {"hospital_orders": {self.agent_id: order_decisions}}

    def _make_order_decisions(self, observation: Dict, epidemic_forecast: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from distributor using OpenAI."""
        decision_type = "order"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        # --- Add printing for raw LLM decision ---
        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")
        # --- End Add ---

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
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
             # --- Add verbose printing for fallback ---
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based assessment & optimal order.[/]")
             # --- End Add ---

             # Fallback: Use criticality assessment and optimal order tool (code15 logic)
             for drug_id in range(num_drugs):
                  inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                  pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  criticality = drug_info.get("criticality_value", 1)

                  # Get demand forecast (can use projected demand or forecast tool output)
                  # Using forecast tool output here as it looks further ahead
                  base_demand_factor = drug_info.get("base_demand", 10) / 1000
                  demand_forecast_tool = [max(0, cases * base_demand_factor) for cases in epidemic_forecast]

                  # Estimate lead time based on disruption risk
                  transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                  base_lead_time = 1 # Dist -> Hosp is short
                  lead_time = base_lead_time + int(round(transport_risk * 3)) # Increase lead time with risk

                  # Call the optimal order tool (verbose printing is inside the base class method)
                  order_qty = self._run_optimal_order_quantity_tool(
                      inventory, pipeline, demand_forecast_tool, lead_time, criticality
                  )
                  order_decisions[drug_id] = order_qty # Store base order quantity from tool

        # --- Apply Rule-Based Adjustments (Code15 Logic) ---
        # These adjustments are applied *after* the LLM's initial suggestion OR the fallback calculation

        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        for drug_id in range(num_drugs): # Ensure we assess all drugs
             # Run assessment tool regardless of LLM success/failure if rules depend on it
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             stockout_hist = observation.get("stockout_history", []) # Already filtered for region in obs
             demand_hist = observation.get("demand_history", []) # Already filtered for region in obs
             unfulfilled = sum(s.get('unfulfilled', 0) for s in stockout_hist)
             total_demand_hist = sum(d.get('demand', 0) for d in demand_hist)

             # Call assessment tool (verbose printing inside base class method)
             situation = self._run_criticality_assessment_tool(
                 drug_info, stockout_hist, unfulfilled, max(1, total_demand_hist) # Avoid division by zero
             )

             # Adjust order based on criticality assessment (using code15 multipliers)
             crit_category = situation.get("category", "")
             multiplier = 1.0
             if crit_category == "Critical Emergency": multiplier = 3.0 # code15 value
             elif crit_category == "Severe Shortage": multiplier = 2.0 # code15 value
             elif crit_category == "Moderate Concern": multiplier = 1.5 # code15 value
             # Add Potential Issue multiplier if desired
             # elif crit_category == "Potential Issue": multiplier = 1.2

             if abs(multiplier - 1.0) > 0.01: # Only apply if multiplier is not 1
                  # --- Add verbose printing for rule ---
                  if self.verbose:
                       self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying criticality assessment multiplier (Category: {crit_category}, Factor: {multiplier:.2f}).[/]")
                       rules_applied_flag = True
                  # --- End Add ---
                  # Apply multiplier to the current order decision (either from LLM or fallback)
                  current_order = order_decisions.get(drug_id, 0) # Get current value or 0 if missing
                  order_decisions[drug_id] = current_order * multiplier


        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             # Only print comparison if rules actually ran and changed something significantly
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else: # Print final decision even if rules didn't change it
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")
        elif self.verbose: # Print the decision even if no rules applied
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in order_decisions.items() if v > 0} # Ensure keys are integers, only return positive


def create_openai_hospital_agent(
    region_id,
    tools,
    openai_integration,
    memory_length=10,
    verbose=True,
    console =None
):
    """Create a hospital agent powered by OpenAI."""
    return HospitalAgent(region_id, tools, openai_integration, memory_length, verbose, console=console)

# --- END OF FILE src/agents/hospital.py ---