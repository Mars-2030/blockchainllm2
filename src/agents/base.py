# --- START OF FILE src/agents/base.py ---

"""
Base class for LLM-powered agents in the pandemic supply chain simulation.
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np

from config import Colors
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class/module
from src.tools import PandemicSupplyChainTools
from src.llm.openai_integration import OpenAILLMIntegration


class OpenAIPandemicLLMAgent:
    """
    Base class for LLM-powered agents using OpenAI API.
    """

    def __init__(
        self,
        agent_type: str,
        agent_id: Any, # Can be int (region) or 0 (manufacturer)
        tools: PandemicSupplyChainTools, # Expect the tools instance
        openai_integration,
        memory_length: int = 10,
        verbose: bool = True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None, # Add blockchain interface
        # Add num_regions, will be set by subclasses if needed (like manufacturer)
        num_regions: Optional[int] = None,
        use_llm: bool = False
    ):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.tools = tools # Store the tools instance
        self.openai = openai_integration
        self.memory_length = memory_length
        self.verbose = verbose
        self.console = console
        self.blockchain = blockchain_interface # Store the blockchain interface instance
        # Store num_regions if provided, crucial for blockchain case query tool
        self.num_regions = num_regions if num_regions is not None else 0
        self.use_llm = use_llm
        self.memory = []
        self.known_disruptions = []
        self.disruption_predictions = {} # Added to store tool output
        
        # --- Check if LLM is requested but integration is missing ---
        if self.use_llm and not self.openai:
             # This shouldn't happen if main.py handles init correctly, but as a safeguard:
             self._print(f"[bold red]WARNING: Agent {self._get_agent_name()} created with use_llm=True but no OpenAI integration provided! Will default to rule-based.[/]")
             self.use_llm = False

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)

    def add_to_memory(self, observation: Dict):
        """Add current observation to memory."""
        # Infer num_regions if not set during init (e.g., for distributor/hospital)
        if self.num_regions == 0 and observation:
             if 'epidemiological_data' in observation and isinstance(observation['epidemiological_data'], dict):
                 # If manu obs, count keys. If D/H, use region_id relation (less reliable)
                 if self.agent_type == "manufacturer":
                     self.num_regions = len(observation['epidemiological_data'])
                 elif 'downstream_projected_demand_summary' in observation: # Check another manu field
                      self.num_regions = len(observation['downstream_projected_demand_summary'])

             # Simple fallback: if it looks like a full observation dict
             elif 'manufacturer' in observation and 'distributors' in observation:
                  if isinstance(observation['distributors'], dict):
                       self.num_regions = len(observation['distributors'])

             if self.num_regions > 0 and self.verbose:
                  self._print(f"[dim]Inferred num_regions = {self.num_regions} for {self._get_agent_name()}[/dim]")

        self.memory.append(observation)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
        # Update known disruptions based on current observation
        if "disruptions" in observation:
             current_disruptions_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d['start_day'])
                                    for d in observation['disruptions'] if isinstance(d, dict)}
             cutoff_day = observation.get('day', 0) - 30
             self.known_disruptions = [d for d in self.known_disruptions
                                      if isinstance(d, dict) and (d.get('end_day', -1) >= observation.get('day', 0) or d.get('start_day', -1) >= cutoff_day)]
             existing_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                              for d in self.known_disruptions if isinstance(d, dict)}
             for d in observation['disruptions']:
                  if isinstance(d, dict):
                    key = (d.get('type'), d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                    if key not in existing_keys:
                        self.known_disruptions.append(d)


    def decide(self, observation: Dict) -> Dict:
        """
        Make a decision based on the agent's mode (LLM or Rule-Based).
        This method should be overridden by subclasses to implement specific logic branching.
        """
        # --- Add basic logging based on mode ---
        mode_str = "[LLM]" if self.use_llm else "[RULE-BASED]"
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]{mode_str} 🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")
        self.add_to_memory(observation)
        # Subclasses will implement the actual decision logic branching based on self.use_llm
        return {}

    def _get_agent_color(self):
        if self.agent_type == "manufacturer": return Colors.MANUFACTURER
        elif self.agent_type == "distributor": return Colors.DISTRIBUTOR
        elif self.agent_type == "hospital": return Colors.HOSPITAL
        return Colors.WHITE

    def _get_agent_name(self):
        if self.agent_type == "manufacturer": return "Manufacturer"
        elif self.agent_type == "distributor": return f"Distributor (Region {self.agent_id})"
        elif self.agent_type == "hospital": return f"Hospital (Region {self.agent_id})"
        return f"Agent {self.agent_id}"

    def _clean_observation_for_prompt(self, observation: Dict, max_len: int = 3500) -> str:
        """Creates a string representation of the observation, removing specific fields for LLM prompt."""
        # Make a deep copy to avoid modifying the original observation used by rules/fallbacks
        try:
            cleaned_obs = json.loads(json.dumps(observation)) # Simple deep copy
        except TypeError as e:
             self._print(f"[red]Error deep copying observation: {e}. Using shallow copy (may have side effects).[/]")
             cleaned_obs = observation.copy()


        # --- REMOVE specific fields we don't want LLM to rely on directly ---
        if 'epidemiological_data' in cleaned_obs:
            for region_id_str in list(cleaned_obs['epidemiological_data'].keys()): # Use list for safe iteration
                if isinstance(cleaned_obs['epidemiological_data'][region_id_str], dict):
                    cleaned_obs['epidemiological_data'][region_id_str].pop('current_cases', None)
                    cleaned_obs['epidemiological_data'][region_id_str].pop('case_trend', None)
                    # Keep projected_demand if present
                    # Remove empty region dicts if nothing left after removing cases/trend
                    if not cleaned_obs['epidemiological_data'][region_id_str]:
                         del cleaned_obs['epidemiological_data'][region_id_str]
            # If the whole epidemiological_data dict is now empty, remove it
            if not cleaned_obs['epidemiological_data']:
                 del cleaned_obs['epidemiological_data']

        # Simplify drug info
        if 'drug_info' in cleaned_obs:
            cleaned_obs['drug_info'] = {k: {'name': v.get('name'), 'crit': v.get('criticality_value'), 'demand_factor': v.get('base_demand')}
                                        for k, v in cleaned_obs.get('drug_info', {}).items()} # Use .get() for safety

        # Simplify history fields - keep fewer items
        history_limit = 5
        for key in ['recent_orders', 'recent_allocations', 'demand_history', 'stockout_history', 'pending_releases']:
            if key in cleaned_obs and isinstance(cleaned_obs[key], list):
                cleaned_obs[key] = cleaned_obs[key][-history_limit:]

        # Round floats recursively
        def round_nested_floats(item):
            if isinstance(item, dict):
                return {k: round_nested_floats(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [round_nested_floats(elem) for elem in item]
            elif isinstance(item, (float, np.floating)):
                try: return round(item, 1)
                except (ValueError, TypeError): return item # Keep original if rounding fails
            elif isinstance(item, (int, np.integer)):
                 return int(item) # Keep ints as ints
            else:
                return item # Keep other types (str, bool, None) as is

        cleaned_obs = round_nested_floats(cleaned_obs)

        # Convert to JSON string, handle truncation
        try:
            json_string = json.dumps(cleaned_obs, indent=None, separators=(',', ':'), default=str, sort_keys=True)
            if len(json_string) > max_len:
                 truncated_json = json_string[:max_len]
                 last_sep = max(truncated_json.rfind(','), truncated_json.rfind('}'), truncated_json.rfind(']'))
                 if last_sep > 0:
                     return truncated_json[:last_sep] + '...[TRUNCATED]"}'
                 else:
                     return truncated_json + '...[TRUNCATED]"}'
            return json_string
        except TypeError as e:
            self._print(f"[yellow]Could not serialize observation to JSON for prompt: {e}. Using basic string representation.[/]")
            fallback_str = f'"day": {observation.get("day")}, "inventories": {observation.get("inventories")}, "...(error serializing)"'
            return "{" + fallback_str[:max_len] + "}"

    def _create_decision_prompt(self, observation: Dict, decision_type: str) -> str:
        """Create a detailed prompt for the OpenAI LLM, noting case data is external."""
        
        # --- Check added for safety, though should be guarded by caller ---
        if not self.use_llm:
             self._print("[yellow]Warning: _create_decision_prompt called when use_llm is False. Returning empty prompt.[/]")
             return ""
        
        current_day = observation.get("day", "N/A")
        agent_name = self._get_agent_name()

        role_desc = f"You are the {agent_name} in a pandemic supply chain simulation."
        # Add role specifics
        if self.agent_type == "manufacturer":
             role_desc += " Your tasks are production planning and allocating drugs to regional distributors."
        elif self.agent_type == "distributor":
             role_desc += f" Your tasks are ordering drugs from the manufacturer and allocating available drugs to the hospital in your Region {self.agent_id}."
        elif self.agent_type == "hospital":
             role_desc += f" Your task is ordering drugs from the distributor in Region {self.agent_id} to meet patient demand."

        # Add task specifics and JSON guidance
        json_guidance = "Respond ONLY with a valid JSON object with string keys (drug IDs, or region IDs if nested) and numerical amounts."
        if decision_type == "production": task = f"Determine optimal production quantity for each drug. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "allocation" and self.agent_type == "manufacturer": task = f"Determine optimal allocation amounts of AVAILABLE drugs to each requesting regional distributor. {json_guidance} Example: {{\"0\": {{\"0\": 100.0, \"1\": 50.0}}, \"1\": {{...}} }}"
        elif decision_type == "allocation" and self.agent_type == "distributor": task = f"Determine optimal allocation amounts of AVAILABLE drugs to the hospital in Region {self.agent_id}. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        elif decision_type == "order" and self.agent_type == "distributor": task = f"Determine optimal order quantity for each drug to request from the manufacturer. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "order" and self.agent_type == "hospital": task = f"Determine optimal order quantity for each drug to request from the distributor. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        else: task = f"Determine the appropriate {decision_type} actions. {json_guidance}"

        # Key considerations - emphasizing external case data
        considerations = f"""
        Consider the following factors based on the provided JSON data:
        - Current day ({current_day}) and overall pandemic timeline.
        - Current inventory levels (own stock, warehouse if applicable). Check available amounts carefully.
        - Inbound and outbound pipeline quantities (drugs in transit).
        - Epidemiological context: Use projected_demand figures provided in the JSON. ***IMPORTANT: Current regional case counts and trends are NOT included in the JSON below. Assume you have access to trusted, up-to-date case counts via an external tool (e.g., blockchain query). Use this external case data primarily for prioritizing allocations or adjusting safety stock multipliers, but base demand QUANTITIES mainly on the provided 'projected_demand' figures.*** React proactively to rising cases (indicated by external data).
        - Drug characteristics (criticality, base demand factor). Prioritize critical drugs.
        - Active and predicted supply chain disruptions (manufacturing, transportation). Increase buffers if risk is high.
        - Recent orders/allocations history (use mainly for context).
        """
        # Add agent/decision specific considerations
        if self.agent_type == "manufacturer":
            if decision_type == "allocation":
                considerations += """
        - Manufacturer Allocation: PRIORITIZE DOWNSTREAM NEEDS. Base allocations on projected demand across regions (see downstream_projected_demand_summary, estimate regional share using projected_demand). Be MORE GENEROUS if downstream inventory (downstream_inventory_summary) is low relative to projected demand OR if trusted case counts (from external tool) are high/rising, especially for critical drugs. Allocate ONLY AVAILABLE inventory from your 'inventories' field. Use fair allocation principles (considering externally obtained trusted cases and criticality) if supply is limited. Adhere to batching schedules (is_batch_processing_day, days_to_next_batch_process).
                """
            elif decision_type == "production":
                 considerations += """
        - Manufacturer Production: PRODUCE PROACTIVELY. Base production heavily on the SUM of projected demand across all regions (downstream_projected_demand_summary) and anticipate future growth indicated by rising trusted case counts (obtained via external tool). Build warehouse buffers ahead of surges, especially for critical drugs or before batch allocation days. Consider production capacity and active disruptions.
                 """
        elif self.agent_type == "distributor":
             if decision_type == "allocation":
                  considerations += f"""
        - Distributor Allocation: Allocate ONLY to your single hospital (Region {self.agent_id}) based on their likely need (use projected_demand). Allocate ONLY AVAILABLE 'inventories'.
                 """
             elif decision_type == "order":
                  considerations += """
        - Distributor Ordering: Aim for sufficient stock to cover lead time demand + safety stock. PRIORITIZE FUTURE DEMAND. Use hospital's projected_demand and consider regional case trends (obtained externally). If cases are rising (externally), order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively. Consider manufacturer lead times and potential transport disruptions.
                 """
        elif self.agent_type == "hospital":
             if decision_type == "order":
                  considerations += """
        - Hospital Ordering: Order to meet patient demand. PRIORITIZE FUTURE DEMAND. Use your projected_demand and consider case trends (obtained externally). If cases are rising (externally), order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively. Consider distributor lead times and potential transport disruptions.
                 """

        final_instruction = "Based *only* on the information provided above in the JSON (and assuming access to trusted external case data), determine the decision. Respond ONLY with the valid JSON object specified in the task, without any additional text, explanations, or markdown formatting."

        observation_summary = self._clean_observation_for_prompt(observation)

        full_prompt = f"""
        {role_desc}

        Current Simulation Day: {current_day}

        Your Task: {task}

        Key Considerations:
        {considerations}

        Current Situation & Data (JSON format, possibly truncated, excludes current cases/trends):
        {observation_summary}

        {final_instruction}
        """
        # self._print(f"DEBUG: Generated Prompt:\n{full_prompt}") # Uncomment for debugging prompts
        return full_prompt

    # --- Tool execution methods ---

    def _run_epidemic_forecast_tool(self, observation: Dict) -> List[float]:
        """Run epidemic forecasting tool. Relies on projected_demand now."""
        tool_name = "Epidemic Forecast (Demand Based)"
        tool_result = [0.0] * 14 # Default forecast length
        try:
            # Manufacturer: Average projected demands across regions/drugs
            if self.agent_type == "manufacturer":
                total_proj_demand = 0.0
                count = 0
                if "downstream_projected_demand_summary" in observation:
                     for drug_id_str, demand in observation["downstream_projected_demand_summary"].items():
                          try: total_proj_demand += float(demand); count += 1
                          except (ValueError, TypeError): continue
                # Use average total projected demand per drug as a basis
                avg_proj_demand_per_drug = (total_proj_demand / count) if count > 0 else 0.0
                # Simple projection: repeat this average
                tool_result = [avg_proj_demand_per_drug] * 14

            # Distributor or Hospital: Use their specific projected demand
            elif "epidemiological_data" in observation:
                # Find the projected_demand dict for this agent's region
                # Note: Obs cleaning now puts proj demand under epi_data[region_id]
                proj_demand_dict = observation["epidemiological_data"].get(str(self.agent_id), {}).get("projected_demand", {})

                avg_proj_demand = 0.0
                count = 0
                for drug_id, demand in proj_demand_dict.items():
                     try: avg_proj_demand += float(demand); count += 1
                     except (ValueError, TypeError): continue
                if count > 0: avg_proj_demand /= count

                tool_result = [avg_proj_demand] * 14 # Repeat average projection

            if self.verbose:
                 print_res = [f"{x:.1f}" for x in tool_result[:7]]
                 self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output (first 7d):[/] {print_res}{'...' if len(tool_result) > 7 else ''}")

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             # tool_result remains default [0.0] * 14
        return tool_result

    def _run_disruption_prediction_tool(self, observation: Dict) -> Dict:
        """Run disruption prediction tool."""
        tool_name = "Disruption Prediction"
        tool_result = {"manufacturing": {}, "transportation": {}}
        try:
             current_day = observation.get("day", 0)
             # Tool uses agent's internal known_disruptions history
             tool_result = self.tools.disruption_prediction_tool(self.known_disruptions, current_day)
             self.disruption_predictions = tool_result # Store the prediction

             if self.verbose:
                  non_zero_preds = {
                      "manufacturing": {k: f"{v:.2f}" for k, v in tool_result.get("manufacturing", {}).items() if v > 0.01},
                      "transportation": {k: f"{v:.2f}" for k, v in tool_result.get("transportation", {}).items() if v > 0.01},
                  }
                  self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output (Prob > 0.01):[/] {non_zero_preds if any(non_zero_preds.values()) else '{}'}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty dict.[/]")
             tool_result = {"manufacturing": {}, "transportation": {}}
             self.disruption_predictions = tool_result # Store empty prediction on error
        return tool_result

    def _run_allocation_priority_tool(self, drug_info: Dict, region_requests: Dict, region_cases: Dict, available: float) -> Dict:
        """Run allocation priority tool (used by manufacturer fallback)."""
        tool_name = "Allocation Priority"
        tool_result = {}
        fallback_result = {}
        try:
            # Ensure keys are integers if needed by the tool
            int_key_requests = {int(k): float(v) for k, v in region_requests.items() if str(k).isdigit() and v is not None}
            # Ensure region_cases keys are integers for the tool
            int_key_cases = {int(k): float(v) for k, v in region_cases.items() if str(k).isdigit() and v is not None}

            valid_drug_info = drug_info if isinstance(drug_info, dict) else {}
            # Call the static method via the tools instance
            tool_result = self.tools.allocation_priority_tool(valid_drug_info, int_key_requests, int_key_cases, max(0.0, float(available)))

            if self.verbose:
                print_res = {k: f"{v:.1f}" for k, v in tool_result.items()}
                self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")

        except Exception as e:
            self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
            # Fallback: Simple equal allocation among requesters
            positive_requests = {r: req for r, req in region_requests.items() if req is not None and float(req) > 0}
            num_positive_requesters = len(positive_requests)
            if num_positive_requesters > 0 and available > 0:
                 equal_share = available / num_positive_requesters
                 fallback_result = {r: min(req, equal_share) for r, req in positive_requests.items()}
            else: fallback_result = {}

            if self.verbose:
                 print_res = {k: f"{v:.1f}" for k, v in fallback_result.items()}
                 self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using fallback allocation:[/]{print_res}")
            tool_result = fallback_result
        # Ensure all originally requesting regions have an entry, even if 0
        final_result = {int(k): 0.0 for k in region_requests}
        final_result.update({int(k): v for k,v in tool_result.items()})
        return final_result


    def _run_optimal_order_quantity_tool(self, inventory: float, pipeline: float, demand_forecast: List[float], lead_time: int = 3, criticality: float = 1.0) -> float:
         """Run optimal order quantity tool."""
         tool_name = "Optimal Order Quantity"
         tool_result = 0.0
         try:
             safety_stock_factor = 1.0 + ((criticality - 1) / 3.0) * 1.0
             safety_stock_factor = max(1.0, safety_stock_factor)
             cleaned_forecast = [float(d) for d in demand_forecast if isinstance(d, (int, float, np.number))]
             # Call the static method via the tools instance
             tool_result = self.tools.optimal_order_quantity_tool(
                 float(inventory), float(pipeline), cleaned_forecast, int(lead_time), float(safety_stock_factor)
                 )

             if self.verbose:
                 self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {tool_result:.1f}")
         except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning 0.0[/]")
             tool_result = 0.0
         return tool_result

    def _run_criticality_assessment_tool(self, drug_info: Dict, stockout_history: List[Dict], unfulfilled: float, total: float) -> Dict:
        """Run criticality assessment tool."""
        tool_name = "Criticality Assessment"
        fallback_assessment = {
                 "drug_name": drug_info.get("name", "Unknown"),
                 "criticality_score": 0, "category": "Error",
                 "stockout_days_recent": len(stockout_history) if isinstance(stockout_history, list) else 0,
                 "unfulfilled_percentage_recent": 0,
                 "recommendations": ["Assessment tool failed."]
             }
        tool_result = fallback_assessment
        try:
            stockout_history = stockout_history if isinstance(stockout_history, list) else []
            # Call the static method via the tools instance
            tool_result = self.tools.criticality_assessment_tool(
                drug_info, stockout_history, float(unfulfilled), max(1.0, float(total))
            )

            if self.verbose:
                print_res = f"Score: {tool_result.get('criticality_score', 'N/A')}, Category: {tool_result.get('category', 'N/A')}"
                self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] {print_res}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] [red]Error - Using fallback assessment.[/]")
             tool_result = fallback_assessment
        return tool_result

    # --- BLOCKCHAIN TOOL RUNNER ---
    def _run_blockchain_regional_cases_tool(self) -> Optional[Dict[int, int]]:
        """
        Runs the tool to query regional case counts from the blockchain.
        Requires self.blockchain and self.num_regions to be set.
        """
        tool_name = "Blockchain Regional Cases"
        tool_result = None
        if not self.blockchain:
             if self.verbose:
                 self._print(f"[{Colors.BLOCKCHAIN}][{Colors.YELLOW}]Skipping {tool_name} tool: Blockchain interface not available.[/]")
             return None # Cannot run without blockchain

        if not self.num_regions or self.num_regions <= 0:
             # Attempt to infer again if needed
             if self.memory:
                 last_obs = self.memory[-1]
                 if 'downstream_projected_demand_summary' in last_obs:
                     self.num_regions = len(last_obs.get('downstream_projected_demand_summary', {}))
                 elif 'distributors' in last_obs: # Check top-level structure
                      self.num_regions = len(last_obs.get('distributors', {}))
             if not self.num_regions or self.num_regions <= 0:
                 self._print(f"[{Colors.RED}]Error running {tool_name} tool: Number of regions unknown or zero for {self._get_agent_name()}.[/]")
                 return None

        try:
            # Call the static tool method, passing the blockchain interface
            tool_result = self.tools.get_blockchain_regional_cases_tool(
                blockchain_interface=self.blockchain,
                num_regions=self.num_regions
            )

            if self.verbose:
                if tool_result is not None:
                    # Check if any region failed (returned 0 when maybe shouldn't have)
                    # Note: The tool itself handles None returns from interface and defaults to 0
                    # So we just report the dictionary we received.
                    print_res = {k: f"{v}" for k, v in tool_result.items()}
                    self._print(f"[{Colors.TOOL_OUTPUT}][TOOL][{Colors.BLOCKCHAIN}]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")
                else:
                     # This case (tool returning None) means the interface wasn't passed or num_regions was invalid.
                     self._print(f"[{Colors.TOOL_OUTPUT}][TOOL][{Colors.BLOCKCHAIN}]{self._get_agent_name()} - {tool_name} Output:[/] [red]Tool execution failed (check interface/num_regions)[/]")

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL][{Colors.BLOCKCHAIN}]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning None[/]")
             tool_result = None # Ensure None is returned on exception

        # Return the dictionary (potentially with 0s for failed regions) or None
        return tool_result


# --- END OF FILE src/agents/base.py ---