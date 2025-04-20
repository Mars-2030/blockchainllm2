# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json # Import json for checking rule changes

class DistributorAgent(OpenAIPandemicLLMAgent):
    """LLM-powered distributor agent using OpenAI."""

    def __init__(self, region_id, tools, openai_integration, num_regions: int, memory_length=10, verbose=True, console=None):
        super().__init__("distributor", region_id, tools, openai_integration, memory_length, verbose, console=console)
        self.num_regions = num_regions # Store the total number of regions

    def decide(self, observation: Dict) -> Dict:
        """Make ordering and allocation decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Still run for context/LLM
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to manufacturer) ---
        # Pass tool outputs for context, even if fallback uses observation data more directly now
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # --- Allocation decisions (to hospital) ---
        # Allocation primarily uses current inventory and LLM/rules, but forecast context can help LLM
        allocation_decisions = self._make_allocation_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using OpenAI, with enhanced fallback and rules."""
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
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based optimal order.[/]")
             # --- End Add ---
             # Fallback: Use rule-based optimal order tool
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                 criticality = drug_info.get("criticality_value", 1)

                 # --- STEP 1 CHANGE START: Use projected_demand for forecast input ---
                 # Distributor needs to forecast demand from its hospital
                 # Use the distributor's observation which includes hospital's projected demand
                 hospital_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                 hospital_projected_demand = max(0, float(hospital_projected_demand)) # Ensure non-negative float

                 # Estimate lead time based on disruption risk
                 # Use region_id (self.agent_id for distributor) to check transport risk
                 transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                 base_lead_time = 3 # Average Manu->Dist lead time assumption
                 lead_time = base_lead_time + int(round(transport_risk * 5)) # Increase lead time with risk
                 lead_time = max(1, lead_time) # Ensure lead time is at least 1

                 # Create a forecast list based on hospital's projected demand.
                 # The enhanced tool will use trend implicitly. (Step 2 Action 1)
                 # Simple baseline: repeat next day's projection.
                 demand_forecast_for_tool = [hospital_projected_demand] * (lead_time + 1) # +1 review period
                 # --- STEP 1 CHANGE END ---

                 # Call the tool (verbose printing is inside the base class method)
                 order_qty = self._run_optimal_order_quantity_tool(
                     inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                 )
                 order_decisions[drug_id] = order_qty


        # Store decisions before rules for comparison
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        # --- Rule-Based Adjustments after LLM/Fallback ---

        # 1. Existing Disruption/Criticality Buffer
        for drug_id in list(order_decisions.keys()): # Use list() for safe iteration
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             criticality = drug_info.get("criticality_value", 1)
             transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
             if criticality >= 3 and transport_risk > 0.4: # If Critical/High and high risk
                  buffer_factor = 1.3
                  if self.verbose:
                       self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption/criticality buffer (factor: {buffer_factor:.2f}).[/]")
                       rules_applied_flag = True
                  order_decisions[drug_id] *= buffer_factor

        # --- STEP 2 CHANGE START: Add Emergency Override based on distributor cover ---
        for drug_id in list(order_decisions.keys()): # Use list() for safe iteration
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            # Use hospital's projected demand as the relevant downstream demand
            hospital_proj_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
            hospital_proj_demand = max(1e-6, float(hospital_proj_demand)) # Ensure positive, avoid div by zero

            inventory_position = inventory + pipeline
            days_cover = inventory_position / hospital_proj_demand

            emergency_boost_factor = 1.0
            # TUNABLE Thresholds:
            if days_cover < 2: # Critically low cover at distributor level
                emergency_boost_factor = 2.0 # Aggressive boost
            elif days_cover < 5: # Moderately low cover
                emergency_boost_factor = 1.3 # Moderate boost

            if abs(emergency_boost_factor - 1.0) > 0.01:
                if self.verbose:
                    self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying low distributor cover EMERGENCY boost (Cover: {days_cover:.1f}d vs Hospital Demand, Factor: {emergency_boost_factor:.2f}).[/]")
                    rules_applied_flag = True
                order_decisions[drug_id] *= emergency_boost_factor
        # --- STEP 2 CHANGE END ---


        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             # Only print comparison if rules changed something
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in order_decisions.items() if v > 0.01} # Ensure keys are integers, filter small amounts



    def _make_allocation_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation to hospital using OpenAI."""
        decision_type = "allocation"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        # --- Add printing for raw LLM decision ---
        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")
        # --- End Add ---

        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, value in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
                      if 0 <= drug_id < num_drugs:
                           alloc_amount = 0.0 # Default to zero

                           # Handle both flat and potentially nested responses (robustness)
                           if isinstance(value, dict):
                               found_numeric = False
                               # Prioritize key '0' or agent_id if present
                               target_keys = [str(self.agent_id), '0']
                               for target_key in target_keys:
                                   if target_key in value:
                                       try:
                                           alloc_amount = max(0.0, float(value[target_key]))
                                           found_numeric = True; break
                                       except (ValueError, TypeError): continue
                               if not found_numeric:
                                   for inner_key, inner_value in value.items():
                                       try:
                                           alloc_amount = max(0.0, float(inner_value))
                                           found_numeric = True; break # Take the first numeric value found
                                       except (ValueError, TypeError): continue
                               if not found_numeric and self.verbose:
                                  self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): LLM nested dict but no numeric value: {value}. Allocating 0.[/]")
                           elif isinstance(value, (int, float, str)):
                              try: alloc_amount = max(0.0, float(value))
                              except (ValueError, TypeError):
                                  if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Cannot convert LLM value '{value}' to float. Allocating 0.[/]")
                           else:
                              if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Unexpected value type from LLM: {type(value)}. Allocating 0.[/]")

                           processed_llm[drug_id] = alloc_amount
                      else:
                           if self.verbose: self._print(f"[{Colors.YELLOW}]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[{Colors.YELLOW}]Error processing {self.agent_type} {decision_type} key '{drug_id_key}' (Region {self.agent_id}): {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm # Start with LLM suggestions

        if not llm_success:
             # --- Add verbose printing for fallback ---
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed or invalid format (Region {self.agent_id}). Using fallback: Fulfill recent order/demand.[/]")
             # --- End Add ---
             # Fallback: Allocate based on recent hospital order or estimated demand
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if inventory <= 0:
                      allocation_decisions[drug_id] = 0
                      continue

                 # Check recent orders from hospital (priority)
                 requested_amount = 0
                 recent_orders = observation.get("recent_orders", [])
                 hospital_id = self.num_regions + 1 + self.agent_id # Calculate relevant hospital ID

                 hospital_orders_for_drug = [o for o in recent_orders if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]
                 if hospital_orders_for_drug:
                     # Sum up recent requests from this hospital for this drug
                     requested_amount = sum(o.get("amount", 0) for o in hospital_orders_for_drug)
                 else:
                     # Fallback: Estimate demand using projected demand from observation
                     # (Hospital projected demand is included in distributor observation)
                     projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                     if projected_demand is not None:
                         requested_amount = max(0, float(projected_demand)) # Ensure float and non-negative
                     else: # Further fallback to case * factor if projected not available
                        drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                        base_demand_factor = drug_info.get("base_demand", 10) / 1000
                        current_cases = observation.get("epidemiological_data", {}).get("current_cases", 0)
                        requested_amount = current_cases * base_demand_factor

                 # Allocate requested amount, capped by inventory (initial cap)
                 allocation_decisions[drug_id] = min(max(0, requested_amount), inventory)


        # --- Apply Final Inventory Cap ---
        # Ensure allocations don't exceed CURRENT inventory AFTER decisions/fallbacks
        final_capped_allocations = {}
        for drug_id, amount in allocation_decisions.items():
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            final_capped_allocations[drug_id] = min(max(0, amount), inventory) # Ensure non-negative and capped


        # --- Add printing for final adjusted decision ---
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_capped_allocations.items()}
             # Check if fallback was used or if decisions changed from raw LLM (even if just capping)
             # This condition might always be true if capping occurs, adjust if needed
             log_prefix = f"[{Colors.DECISION}][FINAL Decision]"
             if not llm_success:
                  log_prefix = f"[{Colors.FALLBACK}][FALLBACK FINAL Decision]"
             elif allocation_decisions != final_capped_allocations:
                  log_prefix = f"[{Colors.RULE}][CAPPED FINAL Decision]" # Indicate capping occurred

             self._print(f"{log_prefix} {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Return integer-keyed dict, only positive allocations
        return {int(k): v for k, v in final_capped_allocations.items() if v > 0.01}


def create_openai_distributor_agent(
    region_id,
    tools,
    openai_integration,
    num_regions: int, # Pass num_regions here
    memory_length=10,
    verbose=True,
    console=None
):
    """Create a distributor agent powered by OpenAI."""
    return DistributorAgent(region_id, tools, openai_integration, num_regions, memory_length, verbose, console=console)

# --- END OF FILE src/agents/distributor.py ---