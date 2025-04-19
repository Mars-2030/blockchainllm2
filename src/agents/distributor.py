# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List
from .base import OpenAIPandemicLLMAgent
from config import Colors


class DistributorAgent(OpenAIPandemicLLMAgent):
    """LLM-powered distributor agent using OpenAI."""

    def __init__(self, region_id, tools, openai_integration, num_regions: int, memory_length=10, verbose=True, console=None):
        super().__init__("distributor", region_id, tools, openai_integration, memory_length, verbose, console=console)
        self.num_regions = num_regions # Store the total number of regions

    def decide(self, observation: Dict) -> Dict:
        """Make ordering and allocation decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        # Tools are run inside the decision methods if needed for fallback or rules
        epidemic_forecast = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to manufacturer) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast, disruption_predictions)

        # --- Allocation decisions (to hospital) ---
        allocation_decisions = self._make_allocation_decisions(observation, epidemic_forecast, disruption_predictions)

        # Structure the output correctly
        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    def _make_order_decisions(self, observation: Dict, epidemic_forecast: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using OpenAI."""
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

                 # Estimate demand forecast for the order tool
                 # Using the epidemic_forecast tool output generated earlier
                 base_demand_factor = drug_info.get("base_demand", 10) / 1000
                 demand_forecast_tool = [max(0, cases * base_demand_factor) for cases in epidemic_forecast]

                 # Estimate lead time based on disruption risk
                 transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                 base_lead_time = 3 # Average Manu->Dist lead time assumption
                 lead_time = base_lead_time + int(round(transport_risk * 5)) # Increase lead time with risk

                 # Call the tool (verbose printing is inside the base class method)
                 order_qty = self._run_optimal_order_quantity_tool(
                     inventory, pipeline, demand_forecast_tool, lead_time, criticality
                 )
                 order_decisions[drug_id] = order_qty

        # Store decisions before rules for comparison
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        # --- Optional Rule-Based Adjustments after LLM/Fallback ---
        # Example: Add buffer for critical drugs if high disruption risk
        for drug_id in order_decisions:
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             criticality = drug_info.get("criticality_value", 1)
             transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
             if criticality >= 3 and transport_risk > 0.4: # If Critical/High and high risk
                  buffer_factor = 1.3
                  # --- Add verbose printing for rule ---
                  if self.verbose:
                       self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption/criticality buffer (factor: {buffer_factor:.2f}).[/]")
                       rules_applied_flag = True
                  # --- End Add ---
                  order_decisions[drug_id] *= buffer_factor # Increase order by 30%


        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in order_decisions.items()} # Ensure keys are integers



    def _make_allocation_decisions(self, observation: Dict, epidemic_forecast: List[float], disruption_predictions: Dict) -> Dict:
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

        # --- Determine num_regions more robustly ---
        #num_regions_in_obs = 0
        # Try getting latest observation from memory which might be the full dict
        # latest_memory_obs = self.memory[-1] if self.memory else {}

        # # Try inferring from the 'distributors' key in the latest memory obs
        # if isinstance(latest_memory_obs.get("distributors"), dict):
        #     num_regions_in_obs = len(latest_memory_obs["distributors"])

        # # Fallback: try inferring from 'hospitals' key
        # if num_regions_in_obs == 0 and isinstance(latest_memory_obs.get("hospitals"), dict):
        #     num_regions_in_obs = len(latest_memory_obs["hospitals"])

        # # Fallback: try inferring from 'epidemiological_data' keys IF they exist at the top level
        # # This assumes the structure { "epidemiological_data": {"0": {...}, "1": {...}} } in memory
        # if num_regions_in_obs == 0 and isinstance(latest_memory_obs.get("epidemiological_data"), dict):
        #      epi_keys = latest_memory_obs["epidemiological_data"].keys()
        #      region_ids_in_epi = {int(k) for k in epi_keys if k.isdigit()}
        #      if region_ids_in_epi:
        #          # Max region ID + 1 = number of regions (assuming 0-indexed)
        #          num_regions_in_obs = max(region_ids_in_epi) + 1

        # # Final fallback if still unknown
        # if num_regions_in_obs == 0:
        #     num_regions_in_obs = 1 # Default guess
        #     if self.verbose: self._print(f"[{Colors.YELLOW}][WARN] Could not reliably determine num_regions for fallback allocation in Dist {self.agent_id}, defaulting to {num_regions_in_obs}. Hospital ID calculation might be incorrect.[/]")
        # --- End Determine num_regions ---


        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, value in structured_decision.items(): # Changed 'amount' to 'value'
                  try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
                      if 0 <= drug_id < num_drugs:
                           alloc_amount = 0.0 # Default to zero

                           # --- Handle both flat and potentially nested responses ---
                           if isinstance(value, dict):
                               found_numeric = False
                               # Prioritize key '0' or agent_id if present, common mistake for single allocation
                               target_keys = [str(self.agent_id), '0']
                               for target_key in target_keys:
                                   if target_key in value:
                                       try:
                                           alloc_amount = max(0.0, float(value[target_key]))
                                           found_numeric = True; break
                                       except (ValueError, TypeError): continue
                               # If not found in prioritized keys, check others
                               if not found_numeric:
                                   for inner_key, inner_value in value.items():
                                       try:
                                           alloc_amount = max(0.0, float(inner_value))
                                           found_numeric = True; break
                                       except (ValueError, TypeError): continue
                               if not found_numeric and self.verbose:
                                  self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): LLM nested dict but no numeric value: {value}. Allocating 0.[/]")
                           elif isinstance(value, (int, float, str)):
                              try: alloc_amount = max(0.0, float(value))
                              except (ValueError, TypeError):
                                  if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Cannot convert LLM value '{value}' to float. Allocating 0.[/]")
                           else:
                              if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Unexpected value type from LLM: {type(value)}. Allocating 0.[/]")
                           # --- End Handle ---

                           processed_llm[drug_id] = alloc_amount # Store LLM suggestion
                      else:
                           if self.verbose: self._print(f"[{Colors.YELLOW}]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[{Colors.YELLOW}]Error processing {self.agent_type} {decision_type} key '{drug_id_key}' (Region {self.agent_id}): {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm # Start with LLM suggestions
                 # Apply inventory cap now
                 for drug_id in allocation_decisions:
                     inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                     allocation_decisions[drug_id] = min(allocation_decisions[drug_id], inventory)

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

                 # Check recent orders from hospital
                 requested_amount = 0
                 recent_orders = observation.get("recent_orders", [])
                 # Calculate hospital_id using inferred num_regions
                 # hospital_id = num_regions_in_obs + 1 + self.agent_id # Use inferred value
                 hospital_id = self.num_regions + 1 + self.agent_id # Use stored value

                 hospital_orders_for_drug = [o for o in recent_orders if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]
                 if hospital_orders_for_drug:
                     requested_amount = sum(o.get("amount", 0) for o in hospital_orders_for_drug)
                 else:
                     # Estimate demand using projected demand from observation
                     projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                     if projected_demand is not None:
                         requested_amount = projected_demand
                     else: # Fallback to case * factor if projected not available
                        drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                        base_demand_factor = drug_info.get("base_demand", 10) / 1000
                        current_cases = observation.get("epidemiological_data", {}).get("current_cases", 0)
                        requested_amount = current_cases * base_demand_factor

                 # Allocate requested amount, capped by inventory
                 allocation_decisions[drug_id] = min(max(0, requested_amount), inventory)

        # --- No specific rules mentioned for distributor allocation, just print final ---
        if self.verbose:
             # Ensure decisions are capped by inventory before printing final
             final_print_decisions = {}
             for drug_id, amount in allocation_decisions.items():
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 final_print_decisions[drug_id] = min(amount, inventory)

             print_after = {k: f"{v:.1f}" for k, v in final_print_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        # Final check: ensure allocations don't exceed inventory before returning
        final_capped_allocations = {}
        for drug_id, amount in allocation_decisions.items():
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            final_capped_allocations[drug_id] = min(amount, inventory)


        return {int(k): v for k, v in final_capped_allocations.items() if v > 0} # Ensure keys are integers, only return positive

def create_openai_distributor_agent(
    region_id,
    tools,
    openai_integration,
    num_regions: int, # Add num_regions here
    memory_length=10,
    verbose=True,
    console=None
):
    """Create a distributor agent powered by OpenAI."""
    return DistributorAgent(region_id, tools, openai_integration, num_regions, memory_length, verbose, console=console)

# --- END OF FILE src/agents/distributor.py ---