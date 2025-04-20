# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List
import numpy as np # Import numpy for calculations
import json # Import json for checking rule changes

from .base import OpenAIPandemicLLMAgent
from config import Colors


class ManufacturerAgent(OpenAIPandemicLLMAgent):
    """LLM-powered manufacturer agent using OpenAI."""

    def __init__(self, tools, openai_integration, memory_length=10, verbose=True, console=None):
        super().__init__("manufacturer", 0, tools, openai_integration, memory_length, verbose, console=console)

    def decide(self, observation: Dict) -> Dict:
        """Make production and allocation decisions using OpenAI."""
        self.add_to_memory(observation) # Store current state

        # --- Use Tools (Run predictions first) ---
        # Epidemic forecast (tool output) might be less critical now for fallback, but useful for LLM context
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Production Decision ---
        production_decisions = self._make_production_decisions(observation, disruption_predictions)

        # --- Allocation Decision ---
        allocation_decisions = self._make_allocation_decisions(observation, disruption_predictions)

        return {
            "manufacturer_production": production_decisions,
            "manufacturer_allocation": allocation_decisions
        }

    def _make_production_decisions(self, observation: Dict, disruption_predictions: Dict) -> Dict:
        """Determine production quantities using OpenAI API, with enhanced rules."""
        decision_type = "production"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        production_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           prod_amount = max(0.0, float(amount))
                           processed_llm[drug_id] = prod_amount
                      else:
                           if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {decision_type} decision.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {decision_type} decision item '{drug_id_key}': {amount} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 production_decisions = processed_llm
                 # Apply capacity cap to initial LLM decisions
                 for drug_id in list(production_decisions.keys()):
                     capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                     production_decisions[drug_id] = min(production_decisions[drug_id], capacity)


        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Max capacity.[/]")
             # Fallback: Produce at max capacity
             for drug_id in range(num_drugs):
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 production_decisions[drug_id] = capacity

        # Store decisions before rules for comparison
        decisions_before_rules = production_decisions.copy()
        rules_applied_flag = False

        # --- Apply Rule-Based Overrides/Adjustments ---

        # 1. Forecasting-based scaling (Existing)
        epidemic_trends = {r: d.get("case_trend", 0) for r, d in observation.get("epidemiological_data", {}).items()}
        growing_regions = sum(1 for trend in epidemic_trends.values() if trend > 0)
        num_regions = len(epidemic_trends)
        if num_regions > 0 and growing_regions > 0:
            production_scale_factor = 1.0 + (growing_regions / num_regions) * 0.5 # Scale up based on proportion
            if self.verbose:
                self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Applying epidemic trend scaling (factor: {production_scale_factor:.2f}).[/]")
                rules_applied_flag = True
            for drug_id in production_decisions:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 scaled_prod = production_decisions[drug_id] * production_scale_factor
                 production_decisions[drug_id] = min(scaled_prod, capacity)

        # 2. Disruption-aware buffer planning (Existing)
        for drug_id in list(production_decisions.keys()):
            disruption_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
            if disruption_risk > 0.1:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 disruption_factor = (1 + 3 * disruption_risk)
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption buffer (risk: {disruption_risk:.2f}, factor: {disruption_factor:.2f}).[/]")
                     rules_applied_flag = True
                 disruption_adjusted_prod = production_decisions[drug_id] * disruption_factor
                 production_decisions[drug_id] = min(disruption_adjusted_prod, capacity)

        # 3. Warehouse Buffer Adjustments (Step 5 - Tuned Factors)
        for drug_id in list(production_decisions.keys()):
            manu_inv = observation.get("inventories", {}).get(str(drug_id), 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(str(drug_id), 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)

            adjustment_factor = 1.0
            if capacity > 0:
                inv_days_cover = total_inv / capacity if capacity > 1 else total_inv
                # --- Step 5 Change: Adjusted thresholds and factors ---
                if inv_days_cover > 7: adjustment_factor = 0.7 # Less reduction (was 0.5 > 5d)
                elif inv_days_cover > 4: adjustment_factor = 0.9 # Less reduction (was 0.7 > 3d)
                elif inv_days_cover < 1.5: adjustment_factor = 1.5 # Keep boost (was < 1d)
                # --- End Step 5 Change ---

            if abs(adjustment_factor - 1.0) > 0.01:
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                     rules_applied_flag = True
                 adjusted_prod = production_decisions[drug_id] * adjustment_factor
                 min_prod = capacity * 0.2
                 production_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # 4. Batch Allocation Awareness (Existing)
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch", 0)
            batch_freq = observation.get("batch_allocation_frequency", 1)
            if batch_freq > 1 and days_to_next_batch <= 2:
                batch_boost_factor = 1.2
                if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Boosting production before batch day (factor: {batch_boost_factor:.2f}).[/]")
                     rules_applied_flag = True
                for drug_id in production_decisions:
                    capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                    batch_boosted_prod = production_decisions[drug_id] * batch_boost_factor
                    production_decisions[drug_id] = min(batch_boosted_prod, capacity)

        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in production_decisions.items()}


    def _make_allocation_decisions(self, observation: Dict, disruption_predictions: Dict) -> Dict:
        """Determine allocation quantities using OpenAI API, with enhanced fallback and proactive rules."""
        decision_type = "allocation"
        prompt = self._create_decision_prompt(observation, decision_type)
        # reasoning = self._simulate_llm_reasoning(prompt) # Optional

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {} # Stores {drug_id: {region_id: amount}}
        num_drugs = len(observation.get("drug_info", {}))
        num_regions = len(observation.get("epidemiological_data", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, region_allocs in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs and isinstance(region_allocs, dict):
                           drug_allocs = {}
                           for region_id_key, amount in region_allocs.items():
                                try:
                                     region_id = int(region_id_key)
                                     if 0 <= region_id < num_regions:
                                          alloc_amount = max(0.0, float(amount))
                                          drug_allocs[region_id] = alloc_amount
                                     else:
                                         if self.verbose: self._print(f"[yellow]Skipping invalid region_id '{region_id_key}' in allocation for Drug {drug_id}.[/]")
                                except (ValueError, TypeError):
                                     if self.verbose: self._print(f"[yellow]Error processing allocation amount for Drug {drug_id}, Region '{region_id_key}': {amount}. Skipping.[/]")
                           if drug_allocs:
                               processed_llm[drug_id] = drug_allocs
                      # elif not isinstance(region_allocs, dict): # Optional: log if value isn't a dict
                      #      if self.verbose: self._print(f"[yellow]Allocation value for Drug {drug_id} is not a dictionary: {region_allocs}. Skipping.[/]")
                      # else: # Optional: log if drug_id invalid
                      #     if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in allocation decision.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing allocation decision item '{drug_id_key}': {region_allocs} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm # Start with LLM suggestions


        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Fair allocation based on projected regional needs.[/]")

             # --- STEP 3 CHANGE START: Improved Fallback Logic ---
             allocation_decisions = {} # Initialize
             lookahead_days = 7 # How many days of demand to cover? (TUNABLE)

             for drug_id in range(num_drugs):
                 available_inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if available_inventory <= 0: continue

                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                 base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000

                 region_needs = {}
                 total_cases_all_regions = sum(d.get("current_cases", 0) for d in observation.get("epidemiological_data", {}).values())

                 # Get total *projected* demand from summary (if available, otherwise estimate)
                 total_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id))

                 if total_proj_demand is not None: # Use summary if available
                     # Distribute total projected demand proportionally to current cases
                     for region_id_str, epi_data in observation.get("epidemiological_data", {}).items():
                          try:
                              region_id = int(region_id_str)
                              current_cases = epi_data.get("current_cases", 0)
                              case_proportion = (current_cases / total_cases_all_regions) if total_cases_all_regions > 0 else (1/num_regions if num_regions > 0 else 0)
                              estimated_daily_regional_demand = total_proj_demand * case_proportion
                              total_need = estimated_daily_regional_demand * lookahead_days
                              region_needs[region_id] = max(0, total_need)
                          except ValueError: continue
                 else: # Fallback to estimating from current cases if summary not present
                      if self.verbose: self._print(f"[yellow]Fallback allocation for Drug {drug_id}: Downstream demand summary missing, estimating from current cases.[/]")
                      for region_id_str, epi_data in observation.get("epidemiological_data", {}).items():
                           try:
                               region_id = int(region_id_str)
                               current_cases = epi_data.get("current_cases", 0)
                               estimated_daily_regional_demand = current_cases * base_demand_per_1k_cases
                               total_need = estimated_daily_regional_demand * lookahead_days
                               region_needs[region_id] = max(0, total_need)
                           except ValueError: continue

                 if not region_needs or available_inventory <= 0: continue

                 # Use the fair allocation logic with calculated needs
                 fair_allocations = self._calculate_fair_allocation(
                     drug_id, region_needs, available_inventory
                 )

                 if fair_allocations:
                    if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                    allocation_decisions[drug_id].update(fair_allocations)
             # --- STEP 3 CHANGE END ---

        # Store decisions before applying rules
        decisions_before_rules = {
            drug_id: regs.copy() for drug_id, regs in allocation_decisions.items()
        } if allocation_decisions else {} # Handle empty initial decisions
        rules_applied_flag = False


        # --- Apply Rule-Based Overrides/Adjustments ---

        # --- STEP 3 CHANGE START: Add Proactive Allocation Rule ---
        # This rule runs *before* batching adjustments to ensure proactivity even off-cycle
        critical_downstream_days = 5 # Threshold (TUNABLE)
        proactive_allocation_factor = 0.3 # Allocate X% of AVAILABLE manu inv if downstream critical (TUNABLE)
        is_overall_trend_positive = any(d.get("case_trend", 0) > 0 for d in observation.get("epidemiological_data", {}).values())

        if is_overall_trend_positive:
            for drug_id_str, summary_data in observation.get("downstream_inventory_summary", {}).items():
                try:
                    drug_id = int(drug_id_str)
                    current_manu_inv = observation.get("inventories", {}).get(drug_id_str, 0)
                    if current_manu_inv <= 0: continue

                    total_downstream_inv = summary_data.get("total_downstream", 0)
                    total_downstream_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str, 0)

                    if total_downstream_proj_demand > 0:
                        days_cover = total_downstream_inv / total_downstream_proj_demand

                        if days_cover < critical_downstream_days:
                            proactive_amount_to_allocate = current_manu_inv * proactive_allocation_factor

                            if proactive_amount_to_allocate > 1:
                                 if self.verbose:
                                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Proactively allocating {proactive_amount_to_allocate:.1f} units ({proactive_allocation_factor*100:.0f}% of available) due to low downstream cover ({days_cover:.1f}d < {critical_downstream_days}d).[/]")
                                     rules_applied_flag = True

                                 # Estimate regional needs again to distribute proactively
                                 region_needs = {}
                                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                                 base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                                 total_cases_all_regions = sum(d.get("current_cases", 0) for d in observation.get("epidemiological_data", {}).values())
                                 total_proj_demand_for_dist = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id), 0)

                                 if total_proj_demand_for_dist is not None and total_cases_all_regions > 0:
                                     for region_id_str_inner, epi_data_inner in observation.get("epidemiological_data", {}).items():
                                          region_id = int(region_id_str_inner)
                                          current_cases_inner = epi_data_inner.get("current_cases", 0)
                                          case_proportion = current_cases_inner / total_cases_all_regions
                                          region_needs[region_id] = max(0, total_proj_demand_for_dist * case_proportion)
                                 else: # Fallback estimation if needed
                                      for region_id_str_inner, epi_data_inner in observation.get("epidemiological_data", {}).items():
                                            region_id = int(region_id_str_inner)
                                            current_cases_inner = epi_data_inner.get("current_cases", 0)
                                            region_needs[region_id] = max(0, current_cases_inner * base_demand_per_1k_cases)


                                 proactive_fair_allocs = self._calculate_fair_allocation(
                                     drug_id, region_needs, proactive_amount_to_allocate
                                 )

                                 if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                                 for region_id, amount in proactive_fair_allocs.items():
                                     current_alloc = allocation_decisions[drug_id].get(region_id, 0)
                                     allocation_decisions[drug_id][region_id] = current_alloc + amount

                except (ValueError, KeyError, TypeError) as e:
                    if self.verbose: self._print(f"[yellow]Warning during proactive allocation rule for drug {drug_id_str}: {e}[/]")
                    continue
        # --- STEP 3 CHANGE END ---


        # Batch Allocation Adjustments (Existing - applied AFTER proactive rule)
        is_batch_day = observation.get("is_batch_day", True) # Default to True if not specified
        batch_freq = observation.get("batch_allocation_frequency", 1)
        if batch_freq > 1 and not is_batch_day: # Apply scale-down only if batching AND not a batch day
             if self.verbose:
                  self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Not a batch day (Freq={batch_freq}d), scaling down non-critical allocations.[/]")
                  rules_applied_flag = True
             for drug_id in list(allocation_decisions.keys()):
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  is_critical = drug_info.get("criticality") == "Critical"
                  if not is_critical:
                     if drug_id in allocation_decisions:
                         for region_id in allocation_decisions[drug_id]:
                             # Apply scaling factor (e.g., 0.25)
                             allocation_decisions[drug_id][region_id] *= 0.25


        # --- Add printing for final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in decisions_before_rules.items()}
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Ensure final decisions use integer keys and filter small amounts
        final_allocations = {}
        for drug_id, allocs in allocation_decisions.items():
            int_allocs = {int(k): v for k, v in allocs.items() if v > 0.01} # Filter small/zero
            if int_allocs: # Only add drug if there are allocations
                 final_allocations[int(drug_id)] = int_allocs
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