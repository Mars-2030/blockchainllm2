# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
import numpy as np # Used in fallback/rules potentially
import json # Import json for checking rule changes

from .base import OpenAIPandemicLLMAgent
from config import Colors

# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Allow running without blockchain deps

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
        """
        Initializes the Distributor agent.

        Args:
            region_id: The ID of the region this distributor serves.
            tools: Instance of PandemicSupplyChainTools.
            openai_integration: Instance of OpenAILLMIntegration.
            num_regions: Total number of regions in the simulation.
            memory_length: How many past observations to remember.
            verbose: Whether to print detailed logs.
            console: Rich console instance for printing.
            blockchain_interface: Optional instance of BlockchainInterface.
        """
        super().__init__(
            agent_type="distributor",
            agent_id=region_id,
            tools=tools,
            openai_integration=openai_integration,
            memory_length=memory_length,
            verbose=verbose,
            console=console,
            blockchain_interface=blockchain_interface, # Pass interface to base
            num_regions=num_regions # Pass num_regions to base
        )
        # Store num_regions specifically if needed for logic (like calculating hospital ID)
        self.num_regions = num_regions

    def decide(self, observation: Dict) -> Dict:
        """Make ordering (from manufacturer) and allocation (to hospital) decisions using OpenAI."""
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")

        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        # Distributor logic primarily relies on hospital projected demand (in obs) and inventory.
        # Blockchain cases are not directly queried by the distributor in this setup.
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Based on hospital demand projection
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (request to manufacturer) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # --- Allocation decisions (shipment to hospital) ---
        allocation_decisions = self._make_allocation_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly for the environment step
        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using OpenAI, with enhanced fallback and rules."""
        decision_type = "order"
        # Prompt uses observation excluding current cases/trends (agents use external BC/projections)
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        # --- Try Parsing LLM Decision ---
        if structured_decision and isinstance(structured_decision, dict):
            processed_llm = {}
            for drug_id_key, amount in structured_decision.items():
                 try:
                     drug_id = int(drug_id_key) # Convert string key from JSON
                     if 0 <= drug_id < num_drugs:
                         order_amount = max(0.0, float(amount)) # Ensure non-negative float
                         processed_llm[drug_id] = order_amount
                     else:
                         if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {self.agent_type} {decision_type} item '{drug_id_key}': {amount} -> {e} (Region {self.agent_id}). Skipping.[/]")

            if processed_llm:
                llm_success = True
                order_decisions = processed_llm # Use LLM decision as starting point

        # --- Fallback Logic if LLM Fails ---
        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based optimal order.[/]")

             order_decisions = {} # Ensure clean slate for fallback
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                 criticality = drug_info.get("criticality_value", 1) # Default to 1 (low)

                 # Use hospital's projected demand from observation as the key driver
                 hospital_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                 hospital_projected_demand = max(0, float(hospital_projected_demand))

                 # Estimate lead time based on disruption risk (Manu -> Dist)
                 # Use combined risk (Transport to region + potential Manu disruption affecting supply)
                 transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                 manu_disrupt_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
                 combined_risk_factor = 1 + (transport_risk * 0.6) + (manu_disrupt_risk * 0.4) # Weighted risk

                 base_lead_time = 3 # Average Manu->Dist lead time assumption (TUNABLE)
                 lead_time = base_lead_time * combined_risk_factor
                 lead_time = max(1, int(round(lead_time))) # Ensure lead time is at least 1 day

                 # Create forecast list based on *hospital's* projected demand.
                 # Assume hospital demand persists; tool uses trend implicitly.
                 demand_forecast_for_tool = [hospital_projected_demand] * (lead_time + 1) # Review period = 1

                 # Call the tool (verbose printing is inside the base class method)
                 order_qty = self._run_optimal_order_quantity_tool(
                     inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                 )
                 order_decisions[drug_id] = order_qty


        # --- Rule-Based Adjustments (Applied AFTER LLM/Fallback) ---
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        # 1. Disruption/Criticality Buffer (TUNED DOWN SLIGHTLY)
        for drug_id in list(order_decisions.keys()): # Use list() for safe iteration
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             criticality = drug_info.get("criticality_value", 1)
             transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
             if criticality >= 3 and transport_risk > 0.5: # If Critical/High and high risk (threshold increased)
                  buffer_factor = 1.2 # WAS 1.3
                  if self.verbose:
                       self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption/criticality buffer (factor: {buffer_factor:.2f}).[/]")
                       rules_applied_flag = True
                  order_decisions[drug_id] *= buffer_factor

        # 2. Emergency Override based on distributor cover vs hospital projected demand (TUNED DOWN)
        for drug_id in list(order_decisions.keys()):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            # Use hospital's projected demand as the relevant downstream demand
            hospital_proj_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
            hospital_proj_demand = max(1e-6, float(hospital_proj_demand)) # Ensure positive, avoid div by zero

            inventory_position = inventory + pipeline
            days_cover = inventory_position / hospital_proj_demand

            emergency_boost_factor = 1.0
            # TUNED Thresholds & Factors:
            if days_cover < 2: # Critically low cover at distributor level
                emergency_boost_factor = 1.5 # WAS 2.0
            elif days_cover < 5: # Moderately low cover
                emergency_boost_factor = 1.1 # WAS 1.3

            if abs(emergency_boost_factor - 1.0) > 0.01:
                if self.verbose:
                    self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying low distributor cover EMERGENCY boost (Cover: {days_cover:.1f}d vs Hospital Demand, Factor: {emergency_boost_factor:.2f}).[/]")
                    rules_applied_flag = True
                # Ensure key exists before multiplying
                order_decisions[drug_id] = order_decisions.get(drug_id, 0.0) * emergency_boost_factor


        # --- Print final adjusted decision ---
        if self.verbose:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             # Only print comparison if rules changed something significantly
             if rules_applied_flag and json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             elif rules_applied_flag: # Rules ran but didn't change output significantly
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
             else: # No rules applied
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        # Ensure keys are integers, filter small/zero amounts
        return {int(k): v for k, v in order_decisions.items() if v > 0.01}


    def _make_allocation_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation quantities to hospital using OpenAI, with fallback and capping."""
        decision_type = "allocation"
        # Prompt uses observation excluding current cases/trends
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        # --- Try Parsing LLM Decision ---
        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, value in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key) # Convert string key from JSON
                      if 0 <= drug_id < num_drugs:
                           alloc_amount = 0.0 # Default to zero

                           # LLM might return value directly or nested under region/hospital ID.
                           # Distributor only allocates to *one* hospital, so we expect a flat value per drug.
                           # We add robustness to handle if LLM nests it anyway.
                           parsed_val = value
                           if isinstance(value, dict):
                               # Calculate the expected hospital ID string
                               hospital_id_str = str(self.num_regions + 1 + self.agent_id)
                               target_keys = ['0', str(self.agent_id), hospital_id_str] # Possible keys

                               found = False
                               for t_key in target_keys:
                                   if t_key in value:
                                       parsed_val = value[t_key]; found = True; break
                               if not found and value: # Fallback: take first value if dict not empty & keys didn't match
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

        # --- Fallback Logic if LLM Fails ---
        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed or invalid format (Region {self.agent_id}). Using fallback: Fulfill recent order / projected demand.[/]")

             allocation_decisions = {} # Ensure clean slate
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

                 # Filter orders *from* this specific hospital for this specific drug
                 hospital_orders_for_drug = [o for o in recent_orders
                                             if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]

                 if hospital_orders_for_drug:
                     # Sum up recent requests (e.g., last order or sum of last few days?) - let's take the latest
                     latest_order_amount = hospital_orders_for_drug[-1].get("amount", 0) if hospital_orders_for_drug else 0
                     requested_amount = latest_order_amount
                     if self.verbose: self._print(f"[dim]Fallback allocation (Drug {drug_id}): Using latest hospital order: {requested_amount:.1f}[/dim]")
                 else:
                     # Fallback: Estimate demand using projected demand from observation
                     projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                     if projected_demand is not None:
                         requested_amount = max(0, float(projected_demand)) # Use next day's projected demand
                         if self.verbose: self._print(f"[dim]Fallback allocation (Drug {drug_id}): No recent order found, using projected demand: {requested_amount:.1f}[/dim]")
                     else:
                         # This case should be less likely now as projected_demand is guaranteed in obs
                         requested_amount = 0
                         if self.verbose: self._print(f"[dim]Fallback allocation (Drug {drug_id}): No recent order or projected demand found. Requesting 0.[/dim]")


                 # Allocate requested amount, capped by inventory (initial cap)
                 allocation_decisions[drug_id] = min(max(0, requested_amount), inventory)


        # --- Apply Final Inventory Cap ---
        # Ensure allocations don't exceed CURRENT inventory AFTER decisions/fallbacks
        final_capped_allocations = {}
        inventory_before_alloc = { # Get snapshot before allocating anything this step
            str(dr_id): observation.get("inventories", {}).get(str(dr_id), 0) for dr_id in range(num_drugs)
        }
        inventory_after_alloc = inventory_before_alloc.copy()

        for drug_id_int, amount in allocation_decisions.items():
            drug_id = str(drug_id_int) # Use string for inventory dict lookup
            # Cap based on inventory available *at this moment in the loop*
            current_available = inventory_after_alloc.get(drug_id, 0.0)
            capped_amount = min(max(0, amount), current_available) # Ensure non-negative and capped
            final_capped_allocations[drug_id_int] = capped_amount
            # Reduce available inventory for subsequent drugs in this step (if needed)
            inventory_after_alloc[drug_id] = current_available - capped_amount


        # --- Print final adjusted decision ---
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_capped_allocations.items()}
             log_prefix = f"[{Colors.DECISION}][FINAL Decision]"
             if not llm_success:
                  log_prefix = f"[{Colors.FALLBACK}][FALLBACK FINAL Decision]"
             # Check if capping actually changed anything compared to the LLM/Fallback decision
             elif allocation_decisions != final_capped_allocations:
                  log_prefix = f"[{Colors.RULE}][CAPPED FINAL Decision]" # Indicate capping occurred

             self._print(f"{log_prefix} {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Return integer-keyed dict, only positive allocations
        return {int(k): v for k, v in final_capped_allocations.items() if v > 0.01}


# --- Factory Function ---

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
    """
    Factory function to create a distributor agent powered by OpenAI.

    Args:
        region_id: The ID of the region this distributor serves.
        tools: Instance of PandemicSupplyChainTools.
        openai_integration: Instance of OpenAILLMIntegration.
        num_regions: Total number of regions in the simulation.
        memory_length: How many past observations to remember.
        verbose: Whether to print detailed logs.
        console: Rich console instance for printing.
        blockchain_interface: Optional instance of BlockchainInterface.

    Returns:
        An initialized DistributorAgent instance.
    """
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