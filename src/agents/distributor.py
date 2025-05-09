# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
Supports LLM and Rule-Based modes.
"""

from typing import Dict, List, Optional
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


class DistributorAgent(OpenAIPandemicLLMAgent):
    """Distributor agent supporting LLM or Rule-Based decisions."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools,
        openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
        num_regions: int,
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None,
        use_llm: bool = False # --- NEW FLAG ---
        ):
        """
        Initializes the Distributor agent.
        """
        super().__init__(
            agent_type="distributor",
            agent_id=region_id,
            tools=tools,
            openai_integration=openai_integration, # Pass Optional integration
            memory_length=memory_length,
            verbose=verbose,
            console=console,
            blockchain_interface=blockchain_interface,
            num_regions=num_regions, # Pass num_regions to base
            use_llm=use_llm # Pass the flag to base
        )
        # Store num_regions explicitly if needed for logic (like calculating hospital ID)
        # self.num_regions is already set in the base class init

    def decide(self, observation: Dict) -> Dict:
        """Make ordering and allocation decisions based on mode (LLM or Rule-Based)."""
        # --- MODIFICATION: Call base decide first for logging/memory ---
        super().decide(observation) # Handles initial logging and memory update

        # --- Use Tools (needed for both modes) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Branch based on mode ---
        if self.use_llm:
            order_decisions = self._make_order_decisions_llm(observation, epidemic_forecast_tool_output, disruption_predictions)
            allocation_decisions = self._make_allocation_decisions_llm(observation, epidemic_forecast_tool_output, disruption_predictions)
        else:
            order_decisions = self._make_order_decisions_rules(observation, epidemic_forecast_tool_output, disruption_predictions)
            allocation_decisions = self._make_allocation_decisions_rules(observation, epidemic_forecast_tool_output, disruption_predictions)

        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    # --- LLM Path for Orders ---
    def _make_order_decisions_llm(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using LLM, with fallback and rules."""
        decision_type = "order"
        agent_name = self._get_agent_name()
        mode_str = "[LLM]"

        prompt = self._create_decision_prompt(observation, decision_type)
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}]{mode_str}[LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
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
                order_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}]{mode_str}[FALLBACK] LLM {agent_name} {decision_type} decision failed/invalid. Using fallback rules.[/]")
             order_decisions = self._get_rule_based_orders(observation, epidemic_forecast_tool_output, disruption_predictions)

        order_decisions = self._apply_order_rules(observation, order_decisions, disruption_predictions)

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}

    # --- Rule-Based Path for Orders ---
    def _make_order_decisions_rules(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities using only rule-based logic."""
        agent_name = self._get_agent_name()
        mode_str = "[RULE-BASED]"
        if self.verbose:
             self._print(f"[{Colors.CYAN}]{mode_str} Calculating {agent_name} order decisions using rules...")

        order_decisions = self._get_rule_based_orders(observation, epidemic_forecast_tool_output, disruption_predictions)
        order_decisions = self._apply_order_rules(observation, order_decisions, disruption_predictions)

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}

    # --- Helper: Calculate initial rule-based orders ---
    def _get_rule_based_orders(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict[int, float]:
        """ Calculates initial order quantities based on rules (optimal order tool). """
        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        for drug_id in range(num_drugs):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
            criticality = drug_info.get("criticality_value", 1)
            hospital_projected_demand = max(0, float(observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)))
            transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
            manu_disrupt_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
            combined_risk_factor = 1 + (transport_risk * 0.6) + (manu_disrupt_risk * 0.4)
            base_lead_time = 3
            lead_time = max(1, int(round(base_lead_time * combined_risk_factor)))
            demand_forecast_for_tool = [hospital_projected_demand] * (lead_time + 1)

            order_qty = self._run_optimal_order_quantity_tool(
                inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
            )
            order_decisions[drug_id] = order_qty
        return order_decisions

    # --- Helper: Apply adjustments to order decisions ---
    def _apply_order_rules(self, observation: Dict, current_orders: Dict, disruption_predictions: Dict) -> Dict[int, float]:
        """ Applies rule-based adjustments (e.g., criticality, low cover) to order decisions. """
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        decision_type = "order"
        decisions_before_rules = current_orders.copy()
        rules_applied_flag = False
        final_decisions = current_orders.copy()

        # 1. Disruption/Criticality Buffer
        for drug_id in list(final_decisions.keys()):
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             criticality = drug_info.get("criticality_value", 1)
             transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
             if criticality >= 3 and transport_risk > 0.5:
                  buffer_factor = 1.2
                  if self.verbose:
                       self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying disruption/criticality buffer (factor: {buffer_factor:.2f}).[/]")
                       rules_applied_flag = True
                  final_decisions[drug_id] *= buffer_factor

        # 2. Emergency Override based on distributor cover vs hospital projected demand
        for drug_id in list(final_decisions.keys()):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            hospital_proj_demand = max(1e-6, float(observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)))
            inventory_position = inventory + pipeline
            days_cover = inventory_position / hospital_proj_demand
            emergency_boost_factor = 1.0
            if days_cover < 2: emergency_boost_factor = 1.5
            elif days_cover < 5: emergency_boost_factor = 1.1
            if abs(emergency_boost_factor - 1.0) > 0.01:
                if self.verbose:
                    self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying low distributor cover EMERGENCY boost (Cover: {days_cover:.1f}d vs Hospital Demand, Factor: {emergency_boost_factor:.2f}).[/]")
                    rules_applied_flag = True
                final_decisions[drug_id] = final_decisions.get(drug_id, 0.0) * emergency_boost_factor

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


    # --- LLM Path for Allocations ---
    def _make_allocation_decisions_llm(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation quantities to hospital using LLM, with fallback and capping."""
        decision_type = "allocation"
        agent_name = self._get_agent_name()
        mode_str = "[LLM]"

        prompt = self._create_decision_prompt(observation, decision_type)
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}]{mode_str}[LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, value in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           alloc_amount = 0.0
                           parsed_val = value
                           if isinstance(value, dict):
                               hospital_id_str = str(self.num_regions + 1 + self.agent_id)
                               target_keys = ['0', str(self.agent_id), hospital_id_str]
                               found = False
                               for t_key in target_keys:
                                   if t_key in value:
                                       parsed_val = value[t_key]; found = True; break
                               if not found and value: parsed_val = next(iter(value.values()))
                           try: alloc_amount = max(0.0, float(parsed_val))
                           except (ValueError, TypeError):
                               if self.verbose: self._print(f"[{Colors.YELLOW}]{mode_str}{agent_name} {decision_type} Drug {drug_id}: Cannot convert LLM value '{parsed_val}' to float. Allocating 0.[/]")
                           processed_llm[drug_id] = alloc_amount
                      else:
                           if self.verbose: self._print(f"[{Colors.YELLOW}]{mode_str}Skipping invalid drug_id '{drug_id_key}' in {agent_name} {decision_type}.[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[{Colors.YELLOW}]{mode_str}Error processing {agent_name} {decision_type} key '{drug_id_key}': {e}. Skipping.[/]")
             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}]{mode_str}[FALLBACK] LLM {agent_name} {decision_type} decision failed or invalid. Using fallback rules.[/]")
             allocation_decisions = self._get_rule_based_allocations(observation)

        # Apply Final Inventory Cap (Rules logic)
        allocation_decisions = self._apply_allocation_rules(observation, allocation_decisions)

        return {int(k): v for k, v in allocation_decisions.items() if v > 0.01}

    # --- Rule-Based Path for Allocations ---
    def _make_allocation_decisions_rules(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation quantities using only rule-based logic."""
        agent_name = self._get_agent_name()
        mode_str = "[RULE-BASED]"
        if self.verbose:
             self._print(f"[{Colors.CYAN}]{mode_str} Calculating {agent_name} allocation decisions using rules...")

        allocation_decisions = self._get_rule_based_allocations(observation)
        allocation_decisions = self._apply_allocation_rules(observation, allocation_decisions) # Apply capping

        return {int(k): v for k, v in allocation_decisions.items() if v > 0.01}

    # --- Helper: Calculate initial rule-based allocations ---
    def _get_rule_based_allocations(self, observation: Dict) -> Dict[int, float]:
        """ Calculates initial allocation quantities based on rules (recent order/projected demand). """
        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]" # Include mode in verbose print

        for drug_id in range(num_drugs):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            if inventory <= 0:
                allocation_decisions[drug_id] = 0
                continue

            requested_amount = 0
            recent_orders = observation.get("recent_orders", [])
            hospital_id = self.num_regions + 1 + self.agent_id
            hospital_orders_for_drug = [o for o in recent_orders if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]

            if hospital_orders_for_drug:
                latest_order_amount = hospital_orders_for_drug[-1].get("amount", 0) if hospital_orders_for_drug else 0
                requested_amount = latest_order_amount
                if self.verbose: self._print(f"[dim]{mode_str}Fallback/Rule allocation (Drug {drug_id}): Using latest hospital order: {requested_amount:.1f}[/dim]")
            else:
                projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                if projected_demand is not None:
                    requested_amount = max(0, float(projected_demand))
                    if self.verbose: self._print(f"[dim]{mode_str}Fallback/Rule allocation (Drug {drug_id}): No recent order, using projected demand: {requested_amount:.1f}[/dim]")
                else:
                    requested_amount = 0
                    if self.verbose: self._print(f"[dim]{mode_str}Fallback/Rule allocation (Drug {drug_id}): No recent order or projected demand. Requesting 0.[/dim]")

            allocation_decisions[drug_id] = max(0, requested_amount) # Request based, cap later

        return allocation_decisions

    # --- Helper: Apply adjustments (capping) to allocation decisions ---
    def _apply_allocation_rules(self, observation: Dict, current_allocations: Dict) -> Dict[int, float]:
        """ Applies rule-based adjustments (inventory capping) to allocation decisions. """
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        decision_type = "allocation"
        final_capped_allocations = {}
        num_drugs = len(observation.get("drug_info", {}))

        # Use a snapshot of inventory before applying any caps for this step
        inventory_snapshot = { str(dr_id): observation.get("inventories", {}).get(str(dr_id), 0) for dr_id in range(num_drugs) }
        inventory_available_for_alloc = inventory_snapshot.copy()
        capped_flag = False

        for drug_id_int, amount in current_allocations.items():
            drug_id = str(drug_id_int)
            current_available = inventory_available_for_alloc.get(drug_id, 0.0)
            capped_amount = min(max(0, amount), current_available)

            if abs(capped_amount - amount) > 0.01 and amount > 0: # Check if capping occurred
                 capped_flag = True

            final_capped_allocations[drug_id_int] = capped_amount
            # Reduce available inventory for this step (though distributor only allocates once)
            inventory_available_for_alloc[drug_id] = current_available - capped_amount

        # Final Logging
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_capped_allocations.items()}
             log_prefix = ""

             if not self.use_llm: # Pure rule-based path
                  log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                  if capped_flag: log_prefix = f"[{Colors.RULE}]{mode_str}[CAPPED FINAL Decision]"
             else: # LLM path (might have included fallback)
                  log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                  if capped_flag: log_prefix = f"[{Colors.RULE}]{mode_str}[CAPPED FINAL Decision]"
                  # Cannot easily detect LLM fallback here without more state

             # Compare capped vs original decision *before* capping
             print_before = {k: f"{v:.1f}" for k, v in current_allocations.items()}
             if capped_flag and json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"{log_prefix} {agent_name} - {decision_type} After Capping:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  cap_status = "(Capped)" if capped_flag else ""
                  self._print(f"{log_prefix} {agent_name} - {decision_type} {cap_status}:[/] {print_after}")

        return final_capped_allocations


# --- MODIFICATION: Update Factory Function ---
def create_openai_distributor_agent(
    region_id,
    tools: PandemicSupplyChainTools,
    openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
    num_regions: int,
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None,
    use_llm: bool = False # --- NEW FLAG ---
) -> DistributorAgent:
    """
    Factory function to create a distributor agent (LLM or Rule-Based).
    """
    return DistributorAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration, # Pass Optional integration
        num_regions=num_regions,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface,
        use_llm=use_llm # Pass the flag
        )

# --- END OF FILE src/agents/distributor.py ---