# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict
from .base import OpenAIPandemicLLMAgent
from config import  Colors


class ManufacturerAgent(OpenAIPandemicLLMAgent):
    """LLM-powered manufacturer agent using OpenAI."""

    def __init__(self, tools, openai_integration, memory_length=10, verbose=True, console=None):
        super().__init__("manufacturer", 0, tools, openai_integration, memory_length, verbose, console=console)

    def decide(self, observation: Dict) -> Dict:
        """Make production and allocation decisions using OpenAI."""
        self.add_to_memory(observation) # Store current state

        # --- Use Tools (Run predictions first) ---
        # Tools are run inside the decision methods if needed for fallback or rules
        # Epidemic forecast is implicitly used via observations/rules
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Production Decision ---
        production_decisions = self._make_production_decisions(observation, disruption_predictions)

        # --- Allocation Decision ---
        # Get allocation decisions (potentially considering production outcomes, though currently decoupled)
        allocation_decisions = self._make_allocation_decisions(observation, disruption_predictions)

        return {
            "manufacturer_production": production_decisions,
            "manufacturer_allocation": allocation_decisions
        }

    def _make_production_decisions(self, observation: Dict, disruption_predictions: Dict) -> Dict:
        """Determine production quantities using OpenAI API."""
        decision_type = "production"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional: Get reasoning text

        # Get structured decision from OpenAI
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        # --- Add printing for raw LLM decision ---
        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")
        # --- End Add ---

        production_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False # Flag to track if LLM provided valid data

        # Process LLM decision or fallback
        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {} # Store successfully processed items
             for drug_id_key, amount in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
                      if 0 <= drug_id < num_drugs:
                           # Ensure amount is float and non-negative
                           prod_amount = max(0.0, float(amount))
                           processed_llm[drug_id] = prod_amount # Use processed_llm to check success
                      else:
                           if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {decision_type} decision.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {decision_type} decision item '{drug_id_key}': {amount} -> {e}. Skipping.[/]")

             if processed_llm: # Check if we got any valid items from LLM
                 llm_success = True
                 production_decisions = processed_llm # Start with LLM suggestions
                 # Apply capacity cap to initial LLM decisions
                 for drug_id in list(production_decisions.keys()):
                     capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                     production_decisions[drug_id] = min(production_decisions[drug_id], capacity)


        if not llm_success: # Fallback triggered if LLM failed completely or returned nothing valid
             # --- Add verbose printing for fallback ---
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Max capacity.[/]")
             # --- End Add ---
             # Fallback: Produce at max capacity (simple strategy)
             for drug_id in range(num_drugs):
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 production_decisions[drug_id] = capacity

        # Store decisions before rules for comparison
        decisions_before_rules = production_decisions.copy()
        rules_applied_flag = False # Track if any rule modified the decision

        # --- Apply Rule-Based Overrides/Adjustments (Code15 Logic) ---

        # Forecasting-based scaling
        epidemic_trends = {r: d.get("case_trend", 0) for r, d in observation.get("epidemiological_data", {}).items()}
        growing_regions = sum(1 for trend in epidemic_trends.values() if trend > 0)
        num_regions = len(epidemic_trends)
        if num_regions > 0 and growing_regions > 0:
            production_scale_factor = 1.0 + (growing_regions / num_regions) * 0.5 # Scale up based on proportion
            # --- Add verbose printing for rule ---
            if self.verbose:
                self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Applying epidemic trend scaling (factor: {production_scale_factor:.2f}).[/]")
                rules_applied_flag = True
            # --- End Add ---
            for drug_id in production_decisions:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 scaled_prod = production_decisions[drug_id] * production_scale_factor
                 production_decisions[drug_id] = min(scaled_prod, capacity) # Cap at capacity

        # Disruption-aware buffer planning
        for drug_id in list(production_decisions.keys()): # Iterate over copy
            disruption_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
            if disruption_risk > 0.1:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 disruption_factor = (1 + 3 * disruption_risk)
                 # --- Add verbose printing for rule ---
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption buffer (risk: {disruption_risk:.2f}, factor: {disruption_factor:.2f}).[/]")
                     rules_applied_flag = True
                 # --- End Add ---
                 disruption_adjusted_prod = production_decisions[drug_id] * disruption_factor
                 production_decisions[drug_id] = min(disruption_adjusted_prod, capacity) # Cap at capacity

        # Warehouse Buffer Adjustments
        for drug_id in list(production_decisions.keys()):
            manu_inv = observation.get("inventories", {}).get(str(drug_id), 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(str(drug_id), 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)

            adjustment_factor = 1.0
            if capacity > 0: # Avoid division by zero
                inv_days_cover = total_inv / capacity if capacity > 1 else total_inv # Rough days cover metric
                if inv_days_cover > 5: adjustment_factor = 0.5
                elif inv_days_cover > 3: adjustment_factor = 0.7
                elif inv_days_cover < 1: adjustment_factor = 1.5

            if abs(adjustment_factor - 1.0) > 0.01: # Only apply if factor is meaningful
                 # --- Add verbose printing for rule ---
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                     rules_applied_flag = True
                 # --- End Add ---
                 adjusted_prod = production_decisions[drug_id] * adjustment_factor
                 # Ensure minimum production (e.g., 20% capacity)
                 min_prod = capacity * 0.2
                 production_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # Batch Allocation Awareness
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch", 0)
            batch_freq = observation.get("batch_allocation_frequency", 1)
            if batch_freq > 1 and days_to_next_batch <= 2: # Approaching batch day
                batch_boost_factor = 1.2
                # --- Add verbose printing for rule ---
                if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Boosting production before batch day (factor: {batch_boost_factor:.2f}).[/]")
                     rules_applied_flag = True
                # --- End Add ---
                for drug_id in production_decisions:
                    capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                    batch_boosted_prod = production_decisions[drug_id] * batch_boost_factor # 20% boost
                    production_decisions[drug_id] = min(batch_boosted_prod, capacity)

        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag: # Only print comparison if rules actually ran
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
        elif self.verbose: # Print the decision even if no rules applied, for clarity
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Ensure final decisions use integer keys
        return {int(k): v for k, v in production_decisions.items()}


    def _make_allocation_decisions(self, observation: Dict, disruption_predictions: Dict) -> Dict:
        """Determine allocation quantities using OpenAI API."""
        decision_type = "allocation"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional

        # Get structured decision from OpenAI
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        # --- Add printing for raw LLM decision ---
        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")
        # --- End Add ---

        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        num_regions = len(observation.get("epidemiological_data", {}))
        llm_success = False # Flag

        # Process LLM decision or fallback
        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, region_allocs in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
                      if 0 <= drug_id < num_drugs:
                           if isinstance(region_allocs, dict):
                                drug_allocs = {}
                                for region_id_key, amount in region_allocs.items():
                                     try:
                                          region_id = int(region_id_key) # Convert string key from JSON
                                          if 0 <= region_id < num_regions:
                                               # Ensure amount is float and non-negative
                                               alloc_amount = max(0.0, float(amount))
                                               drug_allocs[region_id] = alloc_amount
                                          else:
                                              if self.verbose: self._print(f"[yellow]Skipping invalid region_id '{region_id_key}' in allocation for Drug {drug_id}.[/]")
                                     except (ValueError, TypeError):
                                          if self.verbose: self._print(f"[yellow]Error processing allocation amount for Drug {drug_id}, Region '{region_id_key}': {amount}. Skipping.[/]")
                                if drug_allocs: # Only add if we processed at least one region for this drug
                                    processed_llm[drug_id] = drug_allocs
                           else:
                               if self.verbose: self._print(f"[yellow]Allocation value for Drug {drug_id} is not a dictionary: {region_allocs}. Skipping.[/]")
                      else:
                          if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in allocation decision.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing allocation decision item '{drug_id_key}': {region_allocs} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm


        if not llm_success:
             # --- Add verbose printing for fallback ---
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Rule-based fair allocation requests.[/]")
             # --- End Add ---
             # Fallback: Use the environment's detailed fair allocation logic
             allocation_decisions = {} # Will be populated below
             for drug_id in range(num_drugs):
                 # Only manufacturer's *available* inventory can be allocated
                 available_inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if available_inventory <= 0:
                      # No need to add drug_id if no inventory and LLM failed
                      continue

                 # Need requests - Use recent orders as proxy
                 region_requests = {}
                 for order in observation.get("recent_orders", []):
                      if order.get("drug_id") == drug_id:
                           # Order is from distributor 'from_id' = region_id + 1
                           req_region_id = order.get("from_id", -1) # Use -1 to easily check validity
                           if 0 <= req_region_id -1 < num_regions: # Check if valid distributor ID maps to valid region
                                region_id_actual = req_region_id - 1
                                region_requests[region_id_actual] = region_requests.get(region_id_actual, 0) + order.get("amount", 0)

                 # If no recent orders, maybe allocate small amount equally? Or based on cases?
                 if not region_requests and num_regions > 0:
                      # Basic fallback: distribute based on cases if no requests
                      region_cases = {int(r): d.get("current_cases", 0) for r, d in observation.get("epidemiological_data", {}).items() if r.isdigit()}
                      total_cases = sum(region_cases.values())
                      if total_cases > 0:
                           # Allocate small fraction based on cases
                           region_requests = {r: max(0, (c / total_cases) * available_inventory * 0.1)
                                               for r, c in region_cases.items()}
                      else: # If no cases either, allocate tiny amount equally
                           region_requests = {r: max(0, available_inventory / num_regions * 0.05) for r in range(num_regions)}


                 # We'll pass the LLM's suggested allocations (or this fallback) to the step function.
                 # The environment step's _process_manufacturer_allocation will apply the *real* fair allocation
                 # if the sum requested exceeds available inventory.
                 # So, for fallback, we just create a request structure.
                 if drug_id not in allocation_decisions and region_requests: # If LLM failed and we have requests
                     allocation_decisions[drug_id] = {r: amt for r, amt in region_requests.items() if amt > 0} # Only add non-zero requests


        # Store decisions before rules for comparison
        decisions_before_rules = {
            drug_id: regs.copy() for drug_id, regs in allocation_decisions.items()
        }
        rules_applied_flag = False

        # --- Apply Rule-Based Overrides/Adjustments (Code15 Logic) ---

        # Batch Allocation Adjustments (Applied to the *intended* allocation from LLM/fallback)
        is_batch_day = observation.get("is_batch_day", True)
        if not is_batch_day:
             # --- Add verbose printing for rule ---
             if self.verbose:
                  self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Not a batch day, scaling down non-critical allocations.[/]")
                  rules_applied_flag = True # Indicate rule logic was entered
             # --- End Add ---
             for drug_id in list(allocation_decisions.keys()): # Iterate over copy
                  # Check drug criticality - maybe don't scale down critical drugs?
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  is_critical = drug_info.get("criticality") == "Critical"
                  if not is_critical: # Only scale down non-critical on non-batch days
                     if drug_id in allocation_decisions: # Check if drug still exists
                         for region_id in allocation_decisions[drug_id]:
                             allocation_decisions[drug_id][region_id] *= 0.25 # Scale down to 25%


        # Final Check/Log: The environment handles the available inventory constraint via _calculate_fair_allocation.
        # We don't need to scale it down here, just log the intended allocation.

        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in decisions_before_rules.items()}
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             # Only print if there's a notable difference or fallback occurred
             if llm_success is False or json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else: # Print final decision even if rules didn't change it
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")
        elif self.verbose: # Print the decision even if no rules applied
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Ensure final decisions use integer keys
        final_allocations = {}
        for drug_id, allocs in allocation_decisions.items():
            final_allocations[int(drug_id)] = {int(k): v for k, v in allocs.items() if v > 0} # Only keep positive allocations
        # Remove drugs with no allocations
        final_allocations = {k: v for k, v in final_allocations.items() if v}
        return final_allocations


def create_openai_manufacturer_agent(
    tools,
    openai_integration,
    memory_length=10,
    verbose=True,
    console=None
):
    """Create a manufacturer agent powered by OpenAI."""
    return ManufacturerAgent(tools, openai_integration, memory_length, verbose, console=console)

# --- END OF FILE src/agents/manufacturer.py ---