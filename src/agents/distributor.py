# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json # Import json for checking rule changes
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools


class DistributorAgent(OpenAIPandemicLLMAgent):
    """LLM-powered distributor agent using OpenAI."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools, # Expect tools instance
        openai_integration,
        num_regions: int, # Keep num_regions for calculating hospital ID
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add interface
        ):
        super().__init__(
            "distributor",
            region_id,
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
        )
        # Store num_regions if needed for specific logic, e.g., calculating hospital ID
        self.num_regions = num_regions

    def decide(self, observation: Dict) -> Dict:
        """Make ordering and allocation decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        # Distributor *could* query blockchain cases for its region, but the current logic
        # primarily relies on hospital projected demand (provided in observation) and its own inventory.
        # We will not query blockchain cases here for the distributor for now.

        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Now demand-based
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to manufacturer) ---
        # Pass tool outputs for context, even if fallback uses observation data more directly now
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # --- Allocation decisions (to hospital) ---
        # Allocation primarily uses current inventory and LLM/rules, forecast context can help LLM
        allocation_decisions = self._make_allocation_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using OpenAI, with enhanced fallback and rules."""
        decision_type = "order"
        # Prompt uses cleaned observation (no direct cases/trend)
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

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
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based optimal order.[/]")
             # Fallback: Use rule-based optimal order tool (based on projected demand)
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                 criticality = drug_info.get("criticality_value", 1)

                 # Use hospital's projected demand from observation
                 # Ensure the path to projected_demand is correct
                 hospital_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                 hospital_projected_demand = max(0, float(hospital_projected_demand))

                 # Estimate lead time based on disruption risk
                 transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                 base_lead_time = 3 # Average Manu->Dist lead time assumption
                 lead_time = base_lead_time + int(round(transport_risk * 5)) # Increase lead time with risk
                 lead_time = max(1, lead_time) # Ensure lead time is at least 1

                 # Create forecast list based on *hospital's* projected demand.
                 # The enhanced tool will use trend implicitly.
                 demand_forecast_for_tool = [hospital_projected_demand] * (lead_time + 1) # +1 review period

                 # Call the tool (verbose printing is inside the base class method)
                 order_qty = self._run_optimal_order_quantity_tool(
                     inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                 )
                 order_decisions[drug_id] = order_qty


        # Store decisions before rules for comparison
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        # --- Rule-Based Adjustments after LLM/Fallback ---
        # These rules remain valid as they use criticality, risk, and inventory cover relative to projected demand.

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

        # 2. Emergency Override based on distributor cover vs hospital projected demand
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
                # Ensure key exists before multiplying
                if drug_id in order_decisions:
                    order_decisions[drug_id] *= emergency_boost_factor
                else: # Should not happen if loop uses list(keys) but safer
                    order_decisions[drug_id] = 0.0


        # --- Print final adjusted decision ---
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

        # Ensure keys are integers, filter small amounts
        return {int(k): v for k, v in order_decisions.items() if v > 0.01}



    def _make_allocation_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation to hospital using OpenAI."""
        decision_type = "allocation"
        # Prompt uses cleaned observation
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

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

                           # Handle parsing based on expected structure (flat value or nested under region/0)
                           parsed_val = value
                           if isinstance(value, dict):
                               target_keys = ['0', str(self.agent_id)] # Hospital ID is not agent_id here
                               # Hospital ID is num_regions + 1 + region_id (self.agent_id)
                               hospital_id_str = str(self.num_regions + 1 + self.agent_id)
                               target_keys.append(hospital_id_str)

                               found = False
                               for t_key in target_keys:
                                   if t_key in value:
                                       parsed_val = value[t_key]; found = True; break
                               if not found and value: # Fallback: take first value if dict not empty
                                    parsed_val = next(iter(value.values()))

                           try: alloc_amount = max(0.0, float(parsed_val))
                           except (ValueError, TypeError):
                               if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Cannot convert LLM value '{parsed_val}' to float. Allocating 0.[/]")

                           processed_llm[drug_id] = alloc_amount
                      else:
                           if self.verbose: self._print(f"[{Colors.YELLOW}]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[{Colors.YELLOW}]Error processing {self.agent_type} {decision_type} key '{drug_id_key}' (Region {self.agent_id}): {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm # Start with LLM suggestions

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed or invalid format (Region {self.agent_id}). Using fallback: Fulfill recent order/projected demand.[/]")
             # Fallback: Allocate based on recent hospital order or estimated demand
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if inventory <= 0:
                      allocation_decisions[drug_id] = 0
                      continue

                 # Check recent orders from hospital (priority)
                 requested_amount = 0
                 recent_orders = observation.get("recent_orders", [])
                 # Calculate relevant hospital ID based on num_regions and distributor's region_id (agent_id)
                 hospital_id = self.num_regions + 1 + self.agent_id

                 hospital_orders_for_drug = [o for o in recent_orders if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]
                 if hospital_orders_for_drug:
                     # Sum up recent requests from this hospital for this drug
                     requested_amount = sum(o.get("amount", 0) for o in hospital_orders_for_drug)
                 else:
                     # Fallback: Estimate demand using projected demand from observation
                     projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                     if projected_demand is not None:
                         requested_amount = max(0, float(projected_demand)) # Ensure float and non-negative
                     else:
                         # This case should be less likely now as projected_demand is guaranteed in obs
                         requested_amount = 0

                 # Allocate requested amount, capped by inventory (initial cap)
                 allocation_decisions[drug_id] = min(max(0, requested_amount), inventory)


        # --- Apply Final Inventory Cap ---
        # Ensure allocations don't exceed CURRENT inventory AFTER decisions/fallbacks
        final_capped_allocations = {}
        for drug_id, amount in allocation_decisions.items():
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            final_capped_allocations[drug_id] = min(max(0, amount), inventory) # Ensure non-negative and capped


        # --- Print final adjusted decision ---
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_capped_allocations.items()}
             log_prefix = f"[{Colors.DECISION}][FINAL Decision]"
             if not llm_success:
                  log_prefix = f"[{Colors.FALLBACK}][FALLBACK FINAL Decision]"
             # Compare capped vs original decision (could be LLM or fallback)
             elif allocation_decisions != final_capped_allocations:
                  log_prefix = f"[{Colors.RULE}][CAPPED FINAL Decision]" # Indicate capping occurred

             self._print(f"{log_prefix} {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Return integer-keyed dict, only positive allocations
        return {int(k): v for k, v in final_capped_allocations.items() if v > 0.01}


def create_openai_distributor_agent(
    region_id,
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    num_regions: int, # Pass num_regions here
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added interface
):
    """Create a distributor agent powered by OpenAI."""
    return DistributorAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration,
        num_regions=num_regions,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
        )

# --- END OF FILE src/agents/distributor.py ---