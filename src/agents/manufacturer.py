# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
Supports LLM and Rule-Based modes.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import json

from .base import OpenAIPandemicLLMAgent
from config import Colors
# --- MODIFICATION: Optional OpenAI Integration ---
from src.llm.openai_integration import OpenAILLMIntegration
from src.tools import PandemicSupplyChainTools

try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None


class ManufacturerAgent(OpenAIPandemicLLMAgent):
    """Manufacturer agent supporting LLM or Rule-Based decisions."""

    def __init__(
        self,
        tools: PandemicSupplyChainTools,
        openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
        num_regions: int,
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None,
        use_llm: bool = False # --- NEW FLAG ---
    ):
        super().__init__(
            agent_type="manufacturer",
            agent_id=0, # Manufacturer ID is 0
            tools=tools,
            openai_integration=openai_integration, # Pass Optional integration
            memory_length=memory_length,
            verbose=verbose,
            console=console,
            blockchain_interface=blockchain_interface,
            num_regions=num_regions, # Pass num_regions to base
            use_llm=use_llm # Pass the flag to base
        )
        # self.num_regions is set in base class

    def decide(self, observation: Dict) -> Dict:
        """Make production and allocation decisions based on mode (LLM or Rule-Based)."""
        # --- MODIFICATION: Call base decide first for logging/memory ---
        super().decide(observation) # Handles initial logging and memory update

        # --- Query Blockchain (needed for both modes if available) ---
        blockchain_cases = None
        if self.blockchain:
             blockchain_cases = self._run_blockchain_regional_cases_tool()
        if blockchain_cases is None: # Use default if BC disabled or failed
             blockchain_cases = {r: 0 for r in range(self.num_regions)}

        # --- Use Other Tools (needed for both modes) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Branch based on mode ---
        if self.use_llm:
            production_decisions = self._make_production_decisions_llm(observation, disruption_predictions, blockchain_cases)
            allocation_decisions = self._make_allocation_decisions_llm(observation, disruption_predictions, blockchain_cases)
        else:
            production_decisions = self._make_production_decisions_rules(observation, disruption_predictions, blockchain_cases)
            allocation_decisions = self._make_allocation_decisions_rules(observation, disruption_predictions, blockchain_cases)

        return {
            "manufacturer_production": production_decisions,
            "manufacturer_allocation": allocation_decisions
        }

    # --- LLM Path for Production ---
    def _make_production_decisions_llm(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine production quantities using LLM, with fallback and rules."""
        decision_type = "production"
        agent_name = self._get_agent_name()
        mode_str = "[LLM]"

        prompt = self._create_decision_prompt(observation, decision_type)
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}]{mode_str}[LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

        production_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           processed_llm[drug_id] = max(0.0, float(amount))
                      else:
                           if self.verbose: self._print(f"[yellow]{mode_str}Skipping invalid drug_id '{drug_id_key}' in {agent_name} {decision_type}.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]{mode_str}Error processing {agent_name} {decision_type} item '{drug_id_key}': {amount} -> {e}. Skipping.[/]")
             if processed_llm:
                 llm_success = True
                 production_decisions = processed_llm
                 # Apply capacity cap immediately to LLM decisions
                 for drug_id in list(production_decisions.keys()):
                     capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                     production_decisions[drug_id] = min(production_decisions[drug_id], capacity)

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}]{mode_str}[FALLBACK] LLM {agent_name} {decision_type} decision failed/invalid. Using fallback rules.[/]")
             production_decisions = self._get_rule_based_production(observation) # Get initial rule-based decision

        # Apply adjustment rules
        production_decisions = self._apply_production_rules(observation, production_decisions, disruption_predictions, blockchain_cases)

        return {int(k): v for k, v in production_decisions.items()}

    # --- Rule-Based Path for Production ---
    def _make_production_decisions_rules(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine production quantities using only rule-based logic."""
        agent_name = self._get_agent_name()
        mode_str = "[RULE-BASED]"
        if self.verbose:
             self._print(f"[{Colors.CYAN}]{mode_str} Calculating {agent_name} production decisions using rules...")

        production_decisions = self._get_rule_based_production(observation)
        production_decisions = self._apply_production_rules(observation, production_decisions, disruption_predictions, blockchain_cases)

        return {int(k): v for k, v in production_decisions.items()}

    # --- Helper: Calculate initial rule-based production (e.g., max capacity) ---
    def _get_rule_based_production(self, observation: Dict) -> Dict[int, float]:
        """ Calculates initial production quantities based on simple rules (e.g., max capacity). """
        production_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        # Fallback: Produce at max capacity initially, adjustments applied later
        for drug_id in range(num_drugs):
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
            production_decisions[drug_id] = capacity
        return production_decisions

    # --- Helper: Apply adjustments to production decisions ---
    def _apply_production_rules(self, observation: Dict, current_production: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict[int, float]:
        """ Applies rule-based adjustments (scaling, buffers, batching) to production decisions. """
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        decision_type = "production"
        decisions_before_rules = current_production.copy()
        rules_applied_flag = False
        final_decisions = current_production.copy()

        # 1. Scaling based on projected demand and blockchain cases
        total_projected_demand = sum(float(v) for v in observation.get("downstream_projected_demand_summary", {}).values() if v is not None)
        total_capacity = sum(float(c) for c in observation.get("production_capacity", {}).values() if c is not None)
        total_blockchain_cases = sum(blockchain_cases.values())
        production_scale_factor = 1.0
        if total_capacity > 0 and total_projected_demand > total_capacity * 0.8:
            production_scale_factor = 1.2
            if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type}: Scaling up due to high projected demand vs capacity (factor: {production_scale_factor:.2f}).[/]")
        elif total_blockchain_cases > self.num_regions * 500:
             production_scale_factor = 1.1
             if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type}: Scaling up due to high total blockchain cases ({total_blockchain_cases}) (factor: {production_scale_factor:.2f}).[/]")

        if abs(production_scale_factor - 1.0) > 0.01:
             for drug_id in final_decisions:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 scaled_prod = final_decisions[drug_id] * production_scale_factor
                 final_decisions[drug_id] = min(scaled_prod, capacity)

        # 2. Disruption-aware buffer planning
        for drug_id in list(final_decisions.keys()):
            disruption_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
            if disruption_risk > 0.1:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 disruption_factor = (1 + 3 * disruption_risk)
                 if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying disruption buffer (risk: {disruption_risk:.2f}, factor: {disruption_factor:.2f}).[/]")
                 disruption_adjusted_prod = final_decisions[drug_id] * disruption_factor
                 final_decisions[drug_id] = min(disruption_adjusted_prod, capacity)

        # 3. Warehouse Buffer Adjustments
        for drug_id in list(final_decisions.keys()):
            manu_inv = observation.get("inventories", {}).get(str(drug_id), 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(str(drug_id), 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
            adjustment_factor = 1.0
            if capacity > 0:
                proj_demand_summary = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id))
                inv_days_cover = float('inf')
                if proj_demand_summary is not None and proj_demand_summary > 1e-6: inv_days_cover = total_inv / proj_demand_summary
                elif capacity > 1: inv_days_cover = total_inv / capacity
                else: inv_days_cover = total_inv

                if inv_days_cover > 7: adjustment_factor = 0.7
                elif inv_days_cover > 4: adjustment_factor = 0.9
                elif inv_days_cover < 1.5: adjustment_factor = 1.5
                elif inv_days_cover < 3: adjustment_factor = 1.2
            if abs(adjustment_factor - 1.0) > 0.01:
                 if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                 adjusted_prod = final_decisions[drug_id] * adjustment_factor
                 min_prod = capacity * 0.1
                 final_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # 4. Batch Allocation Awareness
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch_process", 0)
            batch_freq = observation.get("batch_allocation_frequency", 1)
            if batch_freq > 1 and days_to_next_batch <= 2:
                batch_boost_factor = 1.3
                if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type}: Boosting production before batch day (factor: {batch_boost_factor:.2f}).[/]")
                for drug_id in final_decisions:
                    capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                    batch_boosted_prod = final_decisions[drug_id] * batch_boost_factor
                    final_decisions[drug_id] = min(batch_boosted_prod, capacity)

        # Final Logging
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_decisions.items()}
             log_prefix = ""
             changed = rules_applied_flag and json.dumps({k: f"{v:.1f}" for k, v in decisions_before_rules.items()}) != json.dumps(print_after)

             if not self.use_llm:
                 log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                 if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
             else:
                  log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                  if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
                  # Cannot easily detect LLM fallback here without more state

             if rules_applied_flag and changed:
                 print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
                 self._print(f"{log_prefix} {agent_name} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                 rule_status = "(Rules checked, no change)" if rules_applied_flag else ""
                 self._print(f"{log_prefix} {agent_name} - {decision_type} {rule_status}:[/] {print_after}")

        return final_decisions


    # --- LLM Path for Allocation ---
    def _make_allocation_decisions_llm(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine allocation quantities using LLM, with fallback and rules."""
        decision_type = "allocation"
        agent_name = self._get_agent_name()
        mode_str = "[LLM]"

        prompt = self._create_decision_prompt(observation, decision_type)
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}]{mode_str}[LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

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
                                     if 0 <= region_id < self.num_regions:
                                          drug_allocs[region_id] = max(0.0, float(amount))
                                     else:
                                         if self.verbose: self._print(f"[yellow]{mode_str}Skipping invalid region_id '{region_id_key}' in allocation for Drug {drug_id}.[/]")
                                except (ValueError, TypeError):
                                     if self.verbose: self._print(f"[yellow]{mode_str}Error processing allocation amount for Drug {drug_id}, Region '{region_id_key}': {amount}. Skipping.[/]")
                           if drug_allocs: processed_llm[drug_id] = drug_allocs
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]{mode_str}Error processing allocation decision item '{drug_id_key}': {region_allocs} -> {e}. Skipping.[/]")
             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm

        if not llm_success:
            if self.verbose:
                self._print(f"[{Colors.FALLBACK}]{mode_str}[FALLBACK] LLM {agent_name} {decision_type} decision failed/invalid. Using fallback rules.[/]")
            allocation_decisions = self._get_rule_based_allocations(observation, blockchain_cases) # Get initial rule-based decision

        # Apply adjustment rules
        allocation_decisions = self._apply_allocation_rules(observation, allocation_decisions, blockchain_cases)

        # Final format check
        final_allocations = {}
        for drug_id, allocs in allocation_decisions.items():
            int_allocs = {int(k): v for k, v in allocs.items() if v > 0.01}
            if int_allocs: final_allocations[int(drug_id)] = int_allocs
        return final_allocations

    # --- Rule-Based Path for Allocation ---
    def _make_allocation_decisions_rules(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine allocation quantities using only rule-based logic."""
        agent_name = self._get_agent_name()
        mode_str = "[RULE-BASED]"
        if self.verbose:
             self._print(f"[{Colors.CYAN}]{mode_str} Calculating {agent_name} allocation decisions using rules...")

        allocation_decisions = self._get_rule_based_allocations(observation, blockchain_cases)
        allocation_decisions = self._apply_allocation_rules(observation, allocation_decisions, blockchain_cases)

        # Final format check
        final_allocations = {}
        for drug_id, allocs in allocation_decisions.items():
            int_allocs = {int(k): v for k, v in allocs.items() if v > 0.01}
            if int_allocs: final_allocations[int(drug_id)] = int_allocs
        return final_allocations

    # --- Helper: Calculate initial rule-based allocations ---
    def _get_rule_based_allocations(self, observation: Dict, blockchain_cases: Dict[int, int]) -> Dict[int, Dict[int, float]]:
        """ Calculates initial allocation quantities based on rules (fair allocation tool). """
        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]" # Include mode in verbose print
        lookahead_days = 7

        for drug_id in range(num_drugs):
            drug_id_str = str(drug_id)
            available_inventory = observation.get("inventories", {}).get(drug_id_str, 0)
            if available_inventory <= 0: continue
            drug_info = observation.get("drug_info", {}).get(drug_id_str, {})

            region_needs = {}
            # --- Prioritize using per-region projected demand ---
            regional_proj_demand = {}
            for r_id_str, epi_data in observation.get("epidemiological_data", {}).items():
                 try:
                     r_id = int(r_id_str)
                     demand = epi_data.get("projected_demand", {}).get(drug_id_str, 0)
                     regional_proj_demand[r_id] = max(0, float(demand))
                 except (ValueError, TypeError): continue
            if regional_proj_demand and sum(regional_proj_demand.values()) > 1e-6:
                if self.verbose: self._print(f"[dim]{mode_str}Fallback/Rule allocation (Drug {drug_id}): Using per-region projected demand.[/dim]")
                for region_id in range(self.num_regions):
                    region_needs[region_id] = max(0, regional_proj_demand.get(region_id, 0) * lookahead_days)
            # --- Fallback 1: Use downstream summary distributed by BC cases ---
            elif "downstream_projected_demand_summary" in observation:
                total_proj_demand_summary = observation["downstream_projected_demand_summary"].get(drug_id_str)
                if total_proj_demand_summary is not None:
                    if self.verbose: self._print(f"[dim]{mode_str}Fallback/Rule allocation (Drug {drug_id}): Using total projected demand distributed by BC cases.[/dim]")
                    total_bc_cases = sum(blockchain_cases.values())
                    for region_id in range(self.num_regions):
                        case_prop = (blockchain_cases.get(region_id, 0) / total_bc_cases) if total_bc_cases > 0 else (1.0 / self.num_regions if self.num_regions > 0 else 0)
                        region_needs[region_id] = max(0, total_proj_demand_summary * case_prop * lookahead_days)
                else: regional_proj_demand = None # Signal absolute fallback
            # --- Fallback 2: Estimate from cases * factor ---
            if not region_needs:
                  if self.verbose: self._print(f"[yellow]{mode_str}Fallback/Rule allocation (Drug {drug_id}): No projected demand found, estimating from blockchain cases * factor.[/]")
                  base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                  for region_id in range(self.num_regions):
                       region_needs[region_id] = max(0, blockchain_cases.get(region_id, 0) * base_demand_per_1k_cases * lookahead_days)

            if not region_needs or available_inventory <= 0: continue

            # Use allocation tool with calculated needs and BC cases for priority
            fair_allocations = self._run_allocation_priority_tool(
                drug_info, region_needs, blockchain_cases, available_inventory
            )
            if fair_allocations:
                int_key_allocs = {int(k): v for k, v in fair_allocations.items()}
                allocation_decisions[drug_id] = int_key_allocs

        return allocation_decisions

    # --- Helper: Apply adjustments to allocation decisions ---
    def _apply_allocation_rules(self, observation: Dict, current_allocations: Dict, blockchain_cases: Dict[int, int]) -> Dict[int, Dict[int, float]]:
        """ Applies rule-based adjustments (proactive, batching) to allocation decisions. """
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        decision_type = "allocation"
        decisions_before_rules = { drug_id: regs.copy() for drug_id, regs in current_allocations.items()}
        rules_applied_flag = False
        final_decisions = current_allocations.copy() # Work on a copy


        # --- Proactive Allocation Rule using Blockchain Cases ---
        critical_downstream_days = 5
        proactive_allocation_factor = 0.3
        total_bc_cases = sum(blockchain_cases.values())
        is_overall_trend_positive = total_bc_cases > self.num_regions * 1000

        for drug_id_str, summary_data in observation.get("downstream_inventory_summary", {}).items():
            try:
                drug_id = int(drug_id_str)
                current_manu_inv = observation.get("inventories", {}).get(drug_id_str, 0)
                if current_manu_inv <= 0: continue

                total_downstream_inv = summary_data.get("total_downstream", 0)
                total_downstream_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str)
                days_cover = float('inf')
                if total_downstream_proj_demand is not None and total_downstream_proj_demand > 1e-6:
                    days_cover = total_downstream_inv / total_downstream_proj_demand

                trigger_proactive = (days_cover < critical_downstream_days) or \
                                    (is_overall_trend_positive and days_cover < critical_downstream_days * 1.5)

                if trigger_proactive:
                    proactive_amount_to_allocate = current_manu_inv * proactive_allocation_factor
                    if proactive_amount_to_allocate > 1:
                        if self.verbose:
                            trigger_reason = f"low downstream cover ({days_cover:.1f}d)" if days_cover < critical_downstream_days else f"high cases ({total_bc_cases}) & moderate cover ({days_cover:.1f}d)"
                            self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Proactively allocating {proactive_amount_to_allocate:.1f} units due to {trigger_reason}.[/]")
                            rules_applied_flag = True

                        # --- Estimate regional needs again for proactive distribution ---
                        region_needs_proactive = {}
                        regional_proj_demand_proactive = {}
                        for r_id_str_p, epi_data_p in observation.get("epidemiological_data", {}).items():
                             try: regional_proj_demand_proactive[int(r_id_str_p)] = max(0, float(epi_data_p.get("projected_demand", {}).get(drug_id_str, 0)))
                             except (ValueError, TypeError): continue
                        if regional_proj_demand_proactive and sum(regional_proj_demand_proactive.values()) > 1e-6:
                            for region_id_inner in range(self.num_regions): region_needs_proactive[region_id_inner] = regional_proj_demand_proactive.get(region_id_inner, 0)
                        else: # Fallback estimation if regional projections are missing/zero
                             total_proj_demand_for_dist = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str, 0)
                             total_bc_cases_inner = sum(blockchain_cases.values())
                             if total_proj_demand_for_dist is not None and total_bc_cases_inner > 0:
                                 for region_id_inner in range(self.num_regions): region_needs_proactive[region_id_inner] = max(0, total_proj_demand_for_dist * (blockchain_cases.get(region_id_inner, 0) / total_bc_cases_inner))
                             else: # Absolute fallback
                                  drug_info_p = observation.get("drug_info", {}).get(drug_id_str, {})
                                  base_demand_per_1k_p = drug_info_p.get("base_demand", 10) / 1000
                                  for region_id_inner in range(self.num_regions): region_needs_proactive[region_id_inner] = max(0, blockchain_cases.get(region_id_inner, 0) * base_demand_per_1k_p)

                        if region_needs_proactive:
                             proactive_fair_allocs = self._run_allocation_priority_tool(
                                 observation.get("drug_info", {}).get(drug_id_str, {}),
                                 region_needs_proactive, blockchain_cases, proactive_amount_to_allocate
                             )
                             if drug_id not in final_decisions: final_decisions[drug_id] = {}
                             for region_id_alloc, amount_alloc in proactive_fair_allocs.items():
                                 region_id_int = int(region_id_alloc)
                                 final_decisions[drug_id][region_id_int] = final_decisions[drug_id].get(region_id_int, 0) + amount_alloc
            except (ValueError, KeyError, TypeError) as e:
                if self.verbose: self._print(f"[yellow]{mode_str}Warning during proactive allocation rule for drug {drug_id_str}: {e}[/]")
                continue


        # --- Batch Allocation Adjustments ---
        is_batch_processing_day = observation.get("is_batch_processing_day", True)
        batch_freq = observation.get("batch_allocation_frequency", 1)
        if batch_freq > 1 and not is_batch_processing_day:
             if self.verbose: rules_applied_flag = True; self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type}: Not a batch day (Freq={batch_freq}d), scaling down non-critical allocations.[/]")
             for drug_id in list(final_decisions.keys()):
                  drug_info_batch = observation.get("drug_info", {}).get(str(drug_id), {})
                  is_critical = drug_info_batch.get("criticality_value", 0) >= 4
                  if not is_critical and drug_id in final_decisions:
                         for region_id in final_decisions[drug_id]:
                             final_decisions[drug_id][region_id] *= 0.25


        # Final Logging
        if self.verbose:
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in final_decisions.items()}
             log_prefix = ""
             changed = rules_applied_flag and json.dumps({dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in decisions_before_rules.items()}) != json.dumps(print_after)

             if not self.use_llm:
                 log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                 if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
             else:
                  log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                  if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
                  # Cannot easily detect LLM fallback here without more state

             if rules_applied_flag and changed:
                 print_before = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in decisions_before_rules.items()}
                 self._print(f"{log_prefix} {agent_name} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                 rule_status = "(Rules checked, no change)" if rules_applied_flag else ""
                 self._print(f"{log_prefix} {agent_name} - {decision_type} {rule_status}:[/] {print_after}")

        return final_decisions


# --- MODIFICATION: Update Factory Function ---
def create_openai_manufacturer_agent(
    tools: PandemicSupplyChainTools,
    openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
    num_regions: int,
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None,
    use_llm: bool = False # --- NEW FLAG ---
):
    """Create a manufacturer agent (LLM or Rule-Based)."""
    return ManufacturerAgent(
        tools=tools,
        openai_integration=openai_integration, # Pass Optional integration
        num_regions=num_regions,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface,
        use_llm=use_llm # Pass the flag
    )

# --- END OF FILE src/agents/manufacturer.py ---