# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
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
            0,
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
        )
        self.num_regions = num_regions # Store number of regions

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
             blockchain_cases = {r: 0 for r in range(self.num_regions)} # Default to 0 if unavailable

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
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 scaled_prod = production_decisions[drug_id] * production_scale_factor
                 production_decisions[drug_id] = min(scaled_prod, capacity)

        # 2. Disruption-aware buffer planning (Existing, unchanged logic)
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

        # 3. Warehouse Buffer Adjustments (Existing, unchanged logic)
        for drug_id in list(production_decisions.keys()):
            manu_inv = observation.get("inventories", {}).get(str(drug_id), 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(str(drug_id), 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)

            adjustment_factor = 1.0
            if capacity > 0:
                inv_days_cover = total_inv / capacity if capacity > 1 else total_inv
                # Adjusted thresholds and factors
                if inv_days_cover > 7: adjustment_factor = 0.7
                elif inv_days_cover > 4: adjustment_factor = 0.9
                elif inv_days_cover < 1.5: adjustment_factor = 1.5

            if abs(adjustment_factor - 1.0) > 0.01:
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                     rules_applied_flag = True
                 adjusted_prod = production_decisions[drug_id] * adjustment_factor
                 min_prod = capacity * 0.2
                 production_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # 4. Batch Allocation Awareness (Existing, uses correct key now)
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch_process", 0) # Use correct key
            batch_freq = observation.get("batch_allocation_frequency", 1)
            if batch_freq > 1 and days_to_next_batch <= 2: # Boost if batch processing is imminent
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
        # num_regions is available as self.num_regions
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

             # --- FALLBACK Logic using Blockchain Cases ---
             allocation_decisions = {} # Initialize
             lookahead_days = 7 # How many days of demand to cover? (TUNABLE)

             for drug_id in range(num_drugs):
                 available_inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if available_inventory <= 0: continue

                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})

                 # Estimate 'need' based on projected demand share, distributed by BC case proportion
                 region_needs = {}
                 total_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id))
                 total_blockchain_cases = sum(blockchain_cases.values())

                 if total_proj_demand is not None:
                     # Use projected demand summary and distribute by case proportion
                     for region_id in range(self.num_regions):
                          current_bc_cases = blockchain_cases.get(region_id, 0)
                          # Calculate proportion safely
                          case_proportion = 0.0
                          if total_blockchain_cases > 0:
                              case_proportion = current_bc_cases / total_blockchain_cases
                          elif self.num_regions > 0: # If no cases, distribute evenly
                               case_proportion = 1.0 / self.num_regions

                          estimated_daily_regional_demand = total_proj_demand * case_proportion
                          total_need = estimated_daily_regional_demand * lookahead_days
                          region_needs[region_id] = max(0, total_need) # Need based on projection share
                 else:
                     # Fallback: Estimate need based purely on cases * demand factor (less ideal)
                      if self.verbose: self._print(f"[yellow]Fallback allocation for Drug {drug_id}: Downstream demand summary missing, estimating from blockchain cases.[/]")
                      base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                      for region_id in range(self.num_regions):
                           current_bc_cases = blockchain_cases.get(region_id, 0)
                           estimated_daily_regional_demand = current_bc_cases * base_demand_per_1k_cases
                           total_need = estimated_daily_regional_demand * lookahead_days
                           region_needs[region_id] = max(0, total_need)

                 if not region_needs or available_inventory <= 0: continue

                 # Call the allocation tool using the blockchain cases for priority
                 # Note: region_needs (based on projection) is passed as 'requests'
                 fair_allocations = self._run_allocation_priority_tool(
                     drug_info, region_needs, blockchain_cases, available_inventory
                 )

                 if fair_allocations:
                    if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                    # Use update to merge dicts correctly
                    allocation_decisions[drug_id].update(fair_allocations)


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
        is_overall_trend_positive = total_bc_cases > self.num_regions * 300 # Trigger if average cases > 300

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
                                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Proactively allocating {proactive_amount_to_allocate:.1f} units due to low downstream cover ({days_cover:.1f}d < {critical_downstream_days}d) & high cases.[/]")
                                     rules_applied_flag = True

                                 # Estimate regional needs again using projected demand share by case proportion
                                 region_needs_proactive = {}
                                 total_proj_demand_for_dist = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id), 0)
                                 total_blockchain_cases_inner = sum(blockchain_cases.values())

                                 if total_proj_demand_for_dist is not None and total_blockchain_cases_inner > 0:
                                     for region_id_inner in range(self.num_regions):
                                          current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                          case_proportion = current_bc_cases_inner / total_blockchain_cases_inner
                                          region_needs_proactive[region_id_inner] = max(0, total_proj_demand_for_dist * case_proportion)
                                 else: # Fallback estimation if needed
                                      drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                                      base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                                      for region_id_inner in range(self.num_regions):
                                           current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                           region_needs_proactive[region_id_inner] = max(0, current_bc_cases_inner * base_demand_per_1k_cases)

                                 # Use the allocation tool with blockchain cases for prioritization
                                 proactive_fair_allocs = self._run_allocation_priority_tool(
                                     observation.get("drug_info", {}).get(str(drug_id), {}),
                                     region_needs_proactive,
                                     blockchain_cases, # Use BC cases for priority
                                     proactive_amount_to_allocate
                                 )

                                 # Add proactive allocation to existing decision
                                 if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                                 for region_id, amount in proactive_fair_allocs.items():
                                     current_alloc = allocation_decisions[drug_id].get(region_id, 0)
                                     allocation_decisions[drug_id][region_id] = current_alloc + amount

                except (ValueError, KeyError, TypeError) as e:
                    if self.verbose: self._print(f"[yellow]Warning during proactive allocation rule for drug {drug_id_str}: {e}[/]")
                    continue


        # Batch Allocation Adjustments (Existing - applied AFTER proactive rule)
        is_batch_day = observation.get("is_batch_processing_day", True) # Use correct key
        batch_freq = observation.get("batch_allocation_frequency", 1)
        if batch_freq > 1 and not is_batch_day: # Apply scale-down only if batching AND not a batch day
             if self.verbose:
                  self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Not a batch day (Freq={batch_freq}d), scaling down non-critical allocations.[/]")
                  rules_applied_flag = True
             for drug_id in list(allocation_decisions.keys()):
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  # Use criticality value for comparison
                  is_critical = drug_info.get("criticality_value", 0) >= 4
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