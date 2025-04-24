# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import json

from .base import OpenAIPandemicLLMAgent
from config import Colors
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools


class ManufacturerAgent(OpenAIPandemicLLMAgent):
    """LLM-powered manufacturer agent using OpenAI."""

    def __init__(
        self,
        tools: PandemicSupplyChainTools,
        openai_integration,
        num_regions: int, # Add num_regions
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add blockchain interface
    ):
        super().__init__(
            "manufacturer",
            0, # Manufacturer ID is 0
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface, # Pass interface to base
            num_regions=num_regions # Pass num_regions to base
        )
        # No need to store num_regions again, it's in self.num_regions from base

    def decide(self, observation: Dict) -> Dict:
        """Make production and allocation decisions using OpenAI."""
        self.add_to_memory(observation) # Store current state

        # --- Query Blockchain for Trusted Data ---
        blockchain_cases = None
        if self.blockchain: # Check if blockchain is enabled for this agent
             # Run the tool to get cases
             blockchain_cases = self._run_blockchain_regional_cases_tool()
             if blockchain_cases is None and self.verbose:
                 self._print(f"[{Colors.FALLBACK}]Manufacturer failed to get cases from Blockchain. Fallback logic will use defaults/projections.[/]")
        # Use a default (e.g., dictionary of zeros) if blockchain query failed or BC is disabled
        if blockchain_cases is None:
             # Ensure default covers the correct number of regions
             blockchain_cases = {r: 0 for r in range(self.num_regions)} # Use self.num_regions

        # --- Use Other Tools (Run predictions first) ---
        # Epidemic forecast tool now relies on projected demand from observation
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Production Decision ---
        # Pass blockchain_cases for potential use in rules/fallback
        production_decisions = self._make_production_decisions(observation, disruption_predictions, blockchain_cases)

        # --- Allocation Decision ---
        # Pass blockchain_cases to allocation decision method
        allocation_decisions = self._make_allocation_decisions(observation, disruption_predictions, blockchain_cases)

        return {
            "manufacturer_production": production_decisions,
            "manufacturer_allocation": allocation_decisions
        }

    def _make_production_decisions(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine production quantities using OpenAI API, with enhanced rules using blockchain cases."""
        decision_type = "production"
        # Prompt now implicitly uses observation where current_cases are removed
        # The prompt instructs the LLM to assume external access to trusted cases
        prompt = self._create_decision_prompt(observation, decision_type)

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
                      # Ensure key is processed as string first for capacity lookup
                      drug_id_str = str(drug_id_key)
                      drug_id = int(drug_id_str) # Then convert to int for logic/return
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
                     drug_id_str = str(drug_id) # Use string key for observation lookup
                     capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)
                     production_decisions[drug_id] = min(production_decisions[drug_id], capacity)


        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Max capacity.[/]")
             # Fallback: Produce at max capacity
             for drug_id in range(num_drugs):
                 drug_id_str = str(drug_id) # Use string key for observation lookup
                 capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)
                 production_decisions[drug_id] = capacity

        # Store decisions before rules for comparison
        decisions_before_rules = production_decisions.copy()
        rules_applied_flag = False

        # --- Apply Rule-Based Overrides/Adjustments ---

        # 1. Scaling based on projected demand and blockchain cases
        # Use total projected demand and total blockchain cases to influence scaling
        total_projected_demand = sum(float(v) for v in observation.get("downstream_projected_demand_summary", {}).values() if v is not None)
        total_capacity = sum(float(c) for c in observation.get("production_capacity", {}).values() if c is not None)
        total_blockchain_cases = sum(blockchain_cases.values())

        production_scale_factor = 1.0
        # Prioritize projected demand for scaling up aggressively
        if total_capacity > 0 and total_projected_demand > total_capacity * 0.8:
            production_scale_factor = 1.2 # Moderate boost if projected demand is high
            if self.verbose:
                self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Scaling up due to high projected demand vs capacity (factor: {production_scale_factor:.2f}).[/]")
                rules_applied_flag = True
        # Also consider scaling based on high absolute case numbers from blockchain
        elif total_blockchain_cases > self.num_regions * 500: # Simple threshold based on average cases per region
             production_scale_factor = 1.1 # Smaller boost if cases are high but projection isn't critical yet
             if self.verbose:
                 self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Scaling up due to high total blockchain cases ({total_blockchain_cases}) (factor: {production_scale_factor:.2f}).[/]")
                 rules_applied_flag = True

        if abs(production_scale_factor - 1.0) > 0.01:
             for drug_id in production_decisions:
                 drug_id_str = str(drug_id) # Use string key for observation lookup
                 capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)
                 scaled_prod = production_decisions[drug_id] * production_scale_factor
                 production_decisions[drug_id] = min(scaled_prod, capacity)

        # 2. Disruption-aware buffer planning
        for drug_id in list(production_decisions.keys()):
            drug_id_str = str(drug_id) # Use string key for observation lookup
            disruption_risk = disruption_predictions.get("manufacturing", {}).get(drug_id_str, 0)
            if disruption_risk > 0.1:
                 capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)
                 disruption_factor = (1 + 3 * disruption_risk) # More aggressive buffer
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption buffer (risk: {disruption_risk:.2f}, factor: {disruption_factor:.2f}).[/]")
                     rules_applied_flag = True
                 disruption_adjusted_prod = production_decisions[drug_id] * disruption_factor
                 production_decisions[drug_id] = min(disruption_adjusted_prod, capacity)

        # 3. Warehouse Buffer Adjustments
        for drug_id in list(production_decisions.keys()):
            drug_id_str = str(drug_id) # Use string key for observation lookup
            manu_inv = observation.get("inventories", {}).get(drug_id_str, 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(drug_id_str, 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)

            adjustment_factor = 1.0
            if capacity > 0:
                # Use projected demand for cover calculation if available, otherwise use capacity
                proj_demand_summary = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str)
                if proj_demand_summary is not None and proj_demand_summary > 1e-6:
                    inv_days_cover = total_inv / proj_demand_summary
                else: # Fallback to capacity if no projection
                    inv_days_cover = total_inv / capacity if capacity > 1 else total_inv

                # Adjusted thresholds and factors based on days cover
                if inv_days_cover > 7: adjustment_factor = 0.7 # Reduce if very high cover
                elif inv_days_cover > 4: adjustment_factor = 0.9 # Slight reduction if high cover
                elif inv_days_cover < 1.5: adjustment_factor = 1.5 # Boost if low cover
                elif inv_days_cover < 3: adjustment_factor = 1.2 # Slight boost if moderately low cover

            if abs(adjustment_factor - 1.0) > 0.01:
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                     rules_applied_flag = True
                 adjusted_prod = production_decisions[drug_id] * adjustment_factor
                 min_prod = capacity * 0.1 # Ensure some minimum production if capacity exists
                 production_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # 4. Batch Allocation Awareness
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch_process", 0)
            batch_freq = observation.get("batch_allocation_frequency", 1)
            # Boost production more significantly if batch processing is imminent (1 or 2 days away)
            if batch_freq > 1 and days_to_next_batch <= 2:
                batch_boost_factor = 1.3 # Slightly increased boost
                if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Boosting production before batch day (factor: {batch_boost_factor:.2f}).[/]")
                     rules_applied_flag = True
                for drug_id in production_decisions:
                    drug_id_str = str(drug_id) # Use string key for observation lookup
                    capacity = observation.get("production_capacity", {}).get(drug_id_str, 0)
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

        return {int(k): v for k, v in production_decisions.items()} # Return with integer keys


    def _make_allocation_decisions(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine allocation quantities using OpenAI API, using blockchain cases for rules/fallback."""
        decision_type = "allocation"
        # Prompt implicitly uses cleaned observation (no direct cases/trend)
        # Instructs LLM to assume external access to trusted cases
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {} # Stores {drug_id: {region_id: amount}}
        num_drugs = len(observation.get("drug_info", {}))
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
                                     # Use self.num_regions for validation
                                     if 0 <= region_id < self.num_regions:
                                          alloc_amount = max(0.0, float(amount))
                                          drug_allocs[region_id] = alloc_amount
                                     else:
                                         if self.verbose: self._print(f"[yellow]Skipping invalid region_id '{region_id_key}' in allocation for Drug {drug_id}.[/]")
                                except (ValueError, TypeError):
                                     if self.verbose: self._print(f"[yellow]Error processing allocation amount for Drug {drug_id}, Region '{region_id_key}': {amount}. Skipping.[/]")
                           if drug_allocs:
                               processed_llm[drug_id] = drug_allocs
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing allocation decision item '{drug_id_key}': {region_allocs} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm # Start with LLM suggestions


        if not llm_success:
            if self.verbose:
                self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Fair allocation based on projected demand & blockchain cases.[/]")

            # --- REVISED FALLBACK Logic using Projected Demand First ---
            allocation_decisions = {} # Initialize
            lookahead_days = 7 # How many days of demand to cover? (TUNABLE)

            for drug_id in range(num_drugs):
                drug_id_str = str(drug_id) # Use string key for observation lookup
                available_inventory = observation.get("inventories", {}).get(drug_id_str, 0)
                if available_inventory <= 0: continue

                drug_info = observation.get("drug_info", {}).get(drug_id_str, {})

                region_needs = {}
                # --- Prioritize using per-region projected demand from observation ---
                regional_proj_demand = {}
                epidemiological_data = observation.get("epidemiological_data", {})
                for r_id_str, epi_data in epidemiological_data.items():
                     try:
                         r_id = int(r_id_str)
                         # Access nested projected_demand dict
                         demand = epi_data.get("projected_demand", {}).get(drug_id_str, 0)
                         regional_proj_demand[r_id] = max(0, float(demand))
                     except (ValueError, TypeError): continue

                # Use regional projections if available and meaningful
                if regional_proj_demand and sum(regional_proj_demand.values()) > 1e-6:
                    if self.verbose: self._print(f"[dim]Fallback allocation for Drug {drug_id}: Using per-region projected demand.[/dim]")
                    for region_id in range(self.num_regions):
                        daily_need = regional_proj_demand.get(region_id, 0)
                        total_need = daily_need * lookahead_days
                        region_needs[region_id] = max(0, total_need) # Need based on projection

                # --- Fallback 1: Use downstream summary distributed by BC cases ---
                elif "downstream_projected_demand_summary" in observation:
                    total_proj_demand_summary = observation["downstream_projected_demand_summary"].get(drug_id_str)
                    if total_proj_demand_summary is not None:
                        if self.verbose: self._print(f"[dim]Fallback allocation for Drug {drug_id}: Using total projected demand distributed by BC cases.[/dim]")
                        total_blockchain_cases = sum(blockchain_cases.values())
                        for region_id in range(self.num_regions):
                            current_bc_cases = blockchain_cases.get(region_id, 0)
                            case_proportion = 0.0
                            if total_blockchain_cases > 0: case_proportion = current_bc_cases / total_blockchain_cases
                            elif self.num_regions > 0: case_proportion = 1.0 / self.num_regions
                            estimated_daily_regional_demand = total_proj_demand_summary * case_proportion
                            total_need = estimated_daily_regional_demand * lookahead_days
                            region_needs[region_id] = max(0, total_need)
                    else: # If summary key exists but value is None, use absolute fallback
                         regional_proj_demand = None # Signal to use absolute fallback

                # --- Fallback 2 (Absolute): Estimate from cases * factor if no projections found ---
                if not region_needs: # If region_needs is still empty after projection checks
                      if self.verbose: self._print(f"[yellow]Fallback allocation for Drug {drug_id}: No projected demand found, estimating from blockchain cases * factor.[/]")
                      base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000 # Use drug's base demand factor
                      for region_id in range(self.num_regions):
                           current_bc_cases = blockchain_cases.get(region_id, 0)
                           estimated_daily_regional_demand = current_bc_cases * base_demand_per_1k_cases
                           total_need = estimated_daily_regional_demand * lookahead_days
                           region_needs[region_id] = max(0, total_need)


                if not region_needs or available_inventory <= 0: continue

                # Call the allocation tool using the calculated region_needs as 'requests'
                # Pass blockchain cases for priority weighting within the tool
                fair_allocations = self._run_allocation_priority_tool(
                    drug_info,
                    region_needs,       # Pass the calculated needs
                    blockchain_cases,   # Pass BC cases for priority weighting
                    available_inventory
                )

                if fair_allocations:
                   if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                   # Use update to merge dicts correctly, ensuring int keys
                   int_key_allocs = {int(k): v for k, v in fair_allocations.items()}
                   allocation_decisions[drug_id].update(int_key_allocs)


        # Store decisions before applying rules
        decisions_before_rules = {
            drug_id: regs.copy() for drug_id, regs in allocation_decisions.items()
        } if allocation_decisions else {} # Handle empty initial decisions
        rules_applied_flag = False


        # --- Apply Rule-Based Overrides/Adjustments ---

        # --- Proactive Allocation Rule using Blockchain Cases ---
        critical_downstream_days = 5 # Threshold (TUNABLE)
        proactive_allocation_factor = 0.3 # Allocate X% of AVAILABLE manu inv if downstream critical (TUNABLE)

        # Determine if trend is positive based on blockchain cases (requires history or comparison)
        # Simple heuristic: boost if total cases are high
        total_bc_cases = sum(blockchain_cases.values())
        # Adjusted threshold to be less sensitive, relying more on downstream cover
        is_overall_trend_positive = total_bc_cases > self.num_regions * 1000

        # Iterate through downstream inventory summary
        for drug_id_str, summary_data in observation.get("downstream_inventory_summary", {}).items():
            try:
                drug_id = int(drug_id_str)
                current_manu_inv = observation.get("inventories", {}).get(drug_id_str, 0)
                if current_manu_inv <= 0: continue # No inventory to proactively allocate

                total_downstream_inv = summary_data.get("total_downstream", 0)
                # Use total projected demand for downstream cover calculation
                total_downstream_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str)

                # Calculate days cover only if projected demand is positive
                days_cover = float('inf') # Default to infinite cover if no demand
                if total_downstream_proj_demand is not None and total_downstream_proj_demand > 1e-6:
                    days_cover = total_downstream_inv / total_downstream_proj_demand

                # Trigger proactive allocation if cover is critically low OR if cases are very high (regardless of cover)
                trigger_proactive = (days_cover < critical_downstream_days) or \
                                    (is_overall_trend_positive and days_cover < critical_downstream_days * 1.5) # Slightly higher cover threshold if cases are high

                if trigger_proactive:
                    proactive_amount_to_allocate = current_manu_inv * proactive_allocation_factor

                    if proactive_amount_to_allocate > 1: # Only allocate meaningful amounts
                        if self.verbose:
                            trigger_reason = f"low downstream cover ({days_cover:.1f}d < {critical_downstream_days}d)" if days_cover < critical_downstream_days else f"high cases ({total_bc_cases}) & moderate cover ({days_cover:.1f}d)"
                            self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Proactively allocating {proactive_amount_to_allocate:.1f} units due to {trigger_reason}.[/]")
                            rules_applied_flag = True

                        # --- Estimate regional needs again using preferred projection method ---
                        region_needs_proactive = {}
                        regional_proj_demand_proactive = {}
                        epidemiological_data_proactive = observation.get("epidemiological_data", {})
                        for r_id_str_proactive, epi_data_proactive in epidemiological_data_proactive.items():
                            try:
                                r_id_proactive = int(r_id_str_proactive)
                                demand_proactive = epi_data_proactive.get("projected_demand", {}).get(drug_id_str, 0)
                                regional_proj_demand_proactive[r_id_proactive] = max(0, float(demand_proactive))
                            except (ValueError, TypeError): continue

                        if regional_proj_demand_proactive and sum(regional_proj_demand_proactive.values()) > 1e-6:
                            for region_id_inner in range(self.num_regions):
                                region_needs_proactive[region_id_inner] = regional_proj_demand_proactive.get(region_id_inner, 0)
                        else: # Fallback estimation if regional projections are missing/zero
                             total_proj_demand_for_dist = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str, 0)
                             total_blockchain_cases_inner = sum(blockchain_cases.values())
                             if total_proj_demand_for_dist is not None and total_blockchain_cases_inner > 0:
                                 for region_id_inner in range(self.num_regions):
                                      current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                      case_proportion = current_bc_cases_inner / total_blockchain_cases_inner
                                      region_needs_proactive[region_id_inner] = max(0, total_proj_demand_for_dist * case_proportion)
                             else: # Absolute fallback
                                  drug_info_proactive = observation.get("drug_info", {}).get(drug_id_str, {})
                                  base_demand_per_1k_cases_proactive = drug_info_proactive.get("base_demand", 10) / 1000
                                  for region_id_inner in range(self.num_regions):
                                       current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                       region_needs_proactive[region_id_inner] = max(0, current_bc_cases_inner * base_demand_per_1k_cases_proactive)


                        # Use the allocation tool with blockchain cases for prioritization
                        if region_needs_proactive: # Only run if needs were calculated
                             proactive_fair_allocs = self._run_allocation_priority_tool(
                                 observation.get("drug_info", {}).get(drug_id_str, {}),
                                 region_needs_proactive,
                                 blockchain_cases, # Use BC cases for priority weighting
                                 proactive_amount_to_allocate # Allocate only the proactively determined amount
                             )

                             # Add proactive allocation to existing decision (ensure int keys)
                             if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                             for region_id_alloc, amount_alloc in proactive_fair_allocs.items():
                                 region_id_int = int(region_id_alloc)
                                 current_alloc = allocation_decisions[drug_id].get(region_id_int, 0)
                                 allocation_decisions[drug_id][region_id_int] = current_alloc + amount_alloc

            except (ValueError, KeyError, TypeError) as e:
                if self.verbose: self._print(f"[yellow]Warning during proactive allocation rule for drug {drug_id_str}: {e}[/]")
                continue


        # Batch Allocation Adjustments (Existing - applied AFTER proactive rule)
        is_batch_processing_day = observation.get("is_batch_processing_day", True)
        batch_freq = observation.get("batch_allocation_frequency", 1)
        if batch_freq > 1 and not is_batch_processing_day: # Apply scale-down only if batching AND not a batch day
             if self.verbose:
                  self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Not a batch day (Freq={batch_freq}d), scaling down non-critical allocations.[/]")
                  rules_applied_flag = True
             for drug_id in list(allocation_decisions.keys()):
                  drug_id_str = str(drug_id) # Use string key for observation lookup
                  drug_info_batch = observation.get("drug_info", {}).get(drug_id_str, {})
                  # Use criticality value for comparison
                  is_critical = drug_info_batch.get("criticality_value", 0) >= 4 # 4 = Critical
                  if not is_critical:
                     if drug_id in allocation_decisions:
                         for region_id in allocation_decisions[drug_id]:
                             # Apply scaling factor (e.g., 0.25) only if not batch day
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
            # Ensure region keys are integers in the final output
            int_allocs = {int(k): v for k, v in allocs.items() if v > 0.01} # Filter small/zero
            if int_allocs: # Only add drug if there are allocations
                 final_allocations[int(drug_id)] = int_allocs
        return final_allocations


def create_openai_manufacturer_agent(
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    num_regions: int, # Added num_regions
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added blockchain interface
):
    """Create a manufacturer agent powered by OpenAI."""
    return ManufacturerAgent(
        tools=tools,
        openai_integration=openai_integration,
        num_regions=num_regions, # Pass num_regions
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
    )

# --- END OF FILE src/agents/manufacturer.py ---