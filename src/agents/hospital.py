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

        # --- Use Tools (Run predictions first, though forecast tool output might be less prioritized now) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Still run for potential LLM context / other rules
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to distributor) ---
        # Pass the tool outputs for context, even if not the primary driver for fallback/rules
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {"hospital_orders": {self.agent_id: order_decisions}}

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from distributor using OpenAI, with enhanced fallback and rules."""
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

             # Fallback: Use criticality assessment and optimal order tool
             for drug_id in range(num_drugs):
                  inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                  pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  criticality = drug_info.get("criticality_value", 1)

                  # --- STEP 1 CHANGE START: Use projected_demand and trend for forecast input ---
                  # Get projected demand for the next day from observation
                  next_day_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                  next_day_projected_demand = max(0, float(next_day_projected_demand)) # Ensure non-negative float

                  # Estimate lead time based on disruption risk
                  # Use region_id (self.agent_id for hospital) to check transport risk
                  transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                  base_lead_time = 1 # Dist -> Hosp is short
                  lead_time = base_lead_time + int(round(transport_risk * 3)) # Increase lead time with risk
                  lead_time = max(1, lead_time) # Ensure lead time is at least 1

                  # Create a forecast list for the tool based on projection.
                  # The enhanced tool will use the trend within this list implicitly (Step 2 Action 1)
                  # If we wanted to pass trend explicitly, we'd calculate it here.
                  # For now, provide a baseline forecast. A better approach might extrapolate using case_trend.
                  # Simplest baseline: repeat next day's projection.
                  # Slight improvement: use the tool's own forecast if available and seems reasonable.
                  # Let's use the tool's forecast generated earlier for simplicity IF it exists.
                  # If not, fall back to the projected demand.
                  demand_forecast_for_tool = []
                  if epidemic_forecast_tool_output:
                      # Need demand factor to convert cases forecast to drug forecast
                      base_demand_factor = drug_info.get("base_demand", 10) / 1000
                      demand_forecast_for_tool = [max(0, cases * base_demand_factor) for cases in epidemic_forecast_tool_output]
                  else:
                      # Fallback if forecast tool failed or wasn't run
                      demand_forecast_for_tool = [next_day_projected_demand] * (lead_time + 1) # +1 review period

                  # Ensure forecast has at least one element if it's empty
                  if not demand_forecast_for_tool:
                       demand_forecast_for_tool = [next_day_projected_demand]

                  # --- STEP 1 CHANGE END ---

                  # Call the optimal order tool (verbose printing is inside the base class method)
                  # It now implicitly handles trend based on the forecast list passed (Step 2 Action 1)
                  order_qty = self._run_optimal_order_quantity_tool(
                      inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                  )
                  order_decisions[drug_id] = order_qty # Store base order quantity from tool


        # --- Apply Rule-Based Adjustments (Step 2 Action 2) ---
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

             # --- STEP 2 CHANGE START: Modified Emergency Override / Multiplier Logic ---
             crit_category = situation.get("category", "")
             base_multiplier = 1.0 # Default

             # Determine base multiplier based on category assessment
             if crit_category == "Critical Emergency": base_multiplier = 3.0
             elif crit_category == "Severe Shortage": base_multiplier = 2.0
             elif crit_category == "Moderate Concern": base_multiplier = 1.5
             # else: base_multiplier remains 1.0 for "Potential Issue" or "Normal"

             # Determine emergency boost target multiplier (overrides base if higher)
             emergency_boost = 1.0 # Default if not emergency/severe
             if crit_category == "Critical Emergency":
                 emergency_boost = 2.5 # More aggressive target boost (TUNABLE)
             elif crit_category == "Severe Shortage":
                 emergency_boost = 1.8 # Moderate boost target (TUNABLE)

             # Use the *highest* relevant multiplier
             final_multiplier = max(base_multiplier, emergency_boost)

             if abs(final_multiplier - 1.0) > 0.01: # Apply if multiplier > 1
                 if self.verbose:
                     reason = "EMERGENCY override boost" if final_multiplier == emergency_boost and emergency_boost > base_multiplier else "criticality assessment multiplier"
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying {reason} (Category: {crit_category}, Factor: {final_multiplier:.2f}).[/]")
                     rules_applied_flag = True
                 current_order = order_decisions.get(drug_id, 0) # Get current value or 0 if missing
                 order_decisions[drug_id] = current_order * final_multiplier
             # --- STEP 2 CHANGE END ---


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
    tools,
    openai_integration,
    memory_length=10,
    verbose=True,
    console =None
):
    """Create a hospital agent powered by OpenAI."""
    return HospitalAgent(region_id, tools, openai_integration, memory_length, verbose, console=console)

# --- END OF FILE src/agents/hospital.py ---