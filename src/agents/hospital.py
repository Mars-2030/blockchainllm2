# --- START OF FILE src/agents/hospital.py ---

"""
Hospital agent implementation for the pandemic supply chain simulation.
Supports LLM and Rule-Based modes.
"""

from typing import Dict, List, Optional
import json
import numpy as np

from .base import OpenAIPandemicLLMAgent
from config import Colors
from src.tools import PandemicSupplyChainTools
# --- MODIFICATION: Optional OpenAI Integration ---
from src.llm.openai_integration import OpenAILLMIntegration

try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None


class HospitalAgent(OpenAIPandemicLLMAgent):
    """Hospital agent supporting LLM or Rule-Based decisions."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools,
        openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None,
        use_llm: bool = False # --- NEW FLAG ---
        ):
        """
        Initializes the Hospital Agent.
        """
        super().__init__(
            agent_type="hospital",
            agent_id=region_id,
            tools=tools,
            openai_integration=openai_integration, # Pass Optional integration
            memory_length=memory_length,
            verbose=verbose,
            console=console,
            blockchain_interface=blockchain_interface,
            use_llm=use_llm # Pass the flag to base
            )
        # No extra storage needed here

    def decide(self, observation: Dict) -> Dict:
        """
        Makes ordering decisions based on mode (LLM or Rule-Based).
        """
        # --- MODIFICATION: Call base decide first for logging/memory ---
        super().decide(observation) # Handles initial logging and memory update

        # --- Use Tools (needed for both modes) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation)
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Branch based on mode ---
        if self.use_llm:
            order_decisions = self._make_order_decisions_llm(observation, epidemic_forecast_tool_output, disruption_predictions)
        else:
            order_decisions = self._make_order_decisions_rules(observation, epidemic_forecast_tool_output, disruption_predictions)

        return {"hospital_orders": {self.agent_id: order_decisions}}

    # --- MODIFICATION: Renamed original method ---
    def _make_order_decisions_llm(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict[int, float]:
        """ Determine order quantities using LLM, with fallback and rules. """
        decision_type = "order"
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" # Explicitly state mode in log

        # Create prompt for LLM
        prompt = self._create_decision_prompt(observation, decision_type)
        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}]{mode_str}[LLM Raw Decision ({agent_name} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        # Process LLM Decision
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

        # Fallback Logic (if LLM failed) - uses rules logic
        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}]{mode_str}[FALLBACK] LLM {agent_name} {decision_type} decision failed/invalid. Using fallback rules.[/]")
             # Call the rule-based logic directly as fallback
             order_decisions = self._get_rule_based_orders(observation, epidemic_forecast_tool_output, disruption_predictions)


        # Apply Rule-Based Adjustments (Applied AFTER LLM/Fallback)
        order_decisions = self._apply_order_rules(observation, order_decisions, disruption_predictions) # Pass current decisions

        # Final Logging (specific to LLM path)
        # Logging of final decision happens within _apply_order_rules now

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}

    # --- MODIFICATION: New method for pure rule-based decisions ---
    def _make_order_decisions_rules(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict[int, float]:
        """ Determine order quantities using only rule-based logic. """
        agent_name = self._get_agent_name()
        mode_str = "[RULE-BASED]"
        if self.verbose:
             self._print(f"[{Colors.CYAN}]{mode_str} Calculating {agent_name} order decisions using rules...")

        # 1. Get initial orders from rule-based logic
        order_decisions = self._get_rule_based_orders(observation, epidemic_forecast_tool_output, disruption_predictions)

        # 2. Apply the same adjustment rules
        order_decisions = self._apply_order_rules(observation, order_decisions, disruption_predictions)

        # Final Logging is handled by _apply_order_rules

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}

    # --- MODIFICATION: Extracted core rule logic for initial calculation ---
    def _get_rule_based_orders(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict[int, float]:
        """ Calculates initial order quantities based on rules (optimal order tool). """
        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))

        for drug_id in range(num_drugs):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
            criticality = drug_info.get("criticality_value", 1)
            next_day_projected_demand = max(0, float(observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)))
            transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
            base_lead_time = 1
            lead_time = max(1, base_lead_time + int(round(transport_risk * 3)))
            planning_horizon = lead_time + 1
            demand_forecast_for_tool = [next_day_projected_demand] * planning_horizon

            order_qty = self._run_optimal_order_quantity_tool(
                inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
            )
            order_decisions[drug_id] = order_qty
        return order_decisions

    # --- MODIFICATION: Extracted rule application logic ---
    def _apply_order_rules(self, observation: Dict, current_orders: Dict, disruption_predictions: Dict) -> Dict[int, float]:
        """ Applies rule-based adjustments (e.g., criticality multipliers) to order decisions. """
        agent_name = self._get_agent_name()
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        decision_type = "order"
        decisions_before_rules = current_orders.copy()
        rules_applied_flag = False
        final_decisions = current_orders.copy() # Work on a copy

        for drug_id in list(final_decisions.keys()):
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             stockout_hist = observation.get("stockout_history", []) or []
             demand_hist = observation.get("demand_history", []) or []
             unfulfilled = sum(s.get('unfulfilled', 0) for s in stockout_hist if isinstance(s, dict))
             total_demand_hist = sum(d.get('demand', 0) for d in demand_hist if isinstance(d, dict))

             situation = self._run_criticality_assessment_tool(
                 drug_info, stockout_hist, unfulfilled, max(1.0, total_demand_hist)
             )

             crit_category = situation.get("category", "Normal Operations")
             final_multiplier = 1.0
             if crit_category == "Critical Emergency": final_multiplier = 2.0
             elif crit_category == "Severe Shortage": final_multiplier = 1.5
             elif crit_category == "Moderate Concern": final_multiplier = 1.2

             if abs(final_multiplier - 1.0) > 0.01:
                 if self.verbose:
                     reason = "criticality assessment multiplier"
                     self._print(f"[{Colors.RULE}]{mode_str}[RULE ADJUSTMENT] {agent_name} - {decision_type} (Drug {drug_id}): Applying {reason} (Category: {crit_category}, Factor: {final_multiplier:.2f}).[/]")
                     rules_applied_flag = True
                 current_order = final_decisions.get(drug_id, 0.0)
                 final_decisions[drug_id] = current_order * final_multiplier

        # Final Logging
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_decisions.items()}
             log_prefix = ""
             changed = rules_applied_flag and json.dumps({k: f"{v:.1f}" for k, v in decisions_before_rules.items()}) != json.dumps(print_after)

             # Determine prefix based on mode and changes
             if not self.use_llm: # Pure rule-based path
                 log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                 if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
             else: # LLM path (might have included fallback)
                  # Use fallback color if initial LLM call failed (check needs to be passed or inferred)
                  # Simplified: assume LLM was attempted if use_llm is True
                  log_prefix = f"[{Colors.DECISION}]{mode_str}[FINAL Decision]"
                  if rules_applied_flag and changed: log_prefix = f"[{Colors.RULE}]{mode_str}[RULE FINAL]"
                  # Note: Cannot easily distinguish here if LLM fallback occurred without passing more state.


             if rules_applied_flag and changed:
                 print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
                 self._print(f"{log_prefix} {agent_name} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                 rule_status = "(Rules checked, no change)" if rules_applied_flag else ""
                 self._print(f"{log_prefix} {agent_name} - {decision_type} {rule_status}:[/] {print_after}")

        return final_decisions


# --- MODIFICATION: Update Factory Function ---
def create_openai_hospital_agent(
    region_id: int,
    tools: PandemicSupplyChainTools,
    openai_integration: Optional[OpenAILLMIntegration], # Accept Optional
    memory_length: int = 10,
    verbose: bool = True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None,
    use_llm: bool = False # --- NEW FLAG ---
) -> HospitalAgent:
    """
    Factory function to create a hospital agent (LLM or Rule-Based).
    """
    return HospitalAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration, # Pass Optional integration
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface,
        use_llm=use_llm # Pass the flag
        )

# --- END OF FILE src/agents/hospital.py ---