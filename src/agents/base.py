# --- START OF FILE src/agents/base.py ---

"""
Base class for LLM-powered agents in the pandemic supply chain simulation.
"""

import json
from typing import Dict, List, Any
import numpy as np

from config import Colors


class OpenAIPandemicLLMAgent:
    """
    Base class for LLM-powered agents using OpenAI API.
    """

    def __init__(
        self,
        agent_type: str,
        agent_id: Any, # Can be int (region) or 0 (manufacturer)
        tools,
        openai_integration,
        memory_length: int = 10,
        verbose: bool = True,
        console=None
    ):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.tools = tools
        self.openai = openai_integration
        self.memory_length = memory_length
        self.verbose = verbose
        self.console = console
        self.memory = []
        self.known_disruptions = []
        self.disruption_predictions = {} # Added to store tool output

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)

    def add_to_memory(self, observation: Dict):
        """Add current observation to memory."""
        self.memory.append(observation)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
        # Update known disruptions based on current observation
        if "disruptions" in observation:
             current_disruptions_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d['start_day']) # Use -1 default for target_id
                                    for d in observation['disruptions'] if isinstance(d, dict)} # Ensure d is a dict
             # Keep known disruptions that are still active or started recently
             cutoff_day = observation.get('day', 0) - 30 # Keep history for 30 days? Adjust as needed
             self.known_disruptions = [d for d in self.known_disruptions
                                      if isinstance(d, dict) and (d.get('end_day', -1) >= observation.get('day', 0) or d.get('start_day', -1) >= cutoff_day)]
             # Add new unique disruptions from current observation
             existing_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                              for d in self.known_disruptions if isinstance(d, dict)}
             for d in observation['disruptions']:
                  if isinstance(d, dict): # Check if d is a dict before accessing keys
                    key = (d.get('type'), d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                    if key not in existing_keys:
                        self.known_disruptions.append(d)


    def decide(self, observation: Dict) -> Dict:
        """Make a decision (placeholder, override in subclasses)."""
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")
        self.add_to_memory(observation)
        # Subclasses will implement the actual decision logic
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

    def _clean_observation_for_prompt(self, observation: Dict, max_len: int = 3500) -> str: # Increased max_len slightly
         """Creates a string representation of the observation, trying to keep it concise."""
         cleaned_obs = observation.copy()

         # Simplify drug info
         if 'drug_info' in cleaned_obs:
             cleaned_obs['drug_info'] = {k: {'name': v.get('name'), 'crit': v.get('criticality_value'), 'demand_factor': v.get('base_demand')}
                                         for k, v in cleaned_obs['drug_info'].items()}

         # Simplify history fields - keep fewer items
         history_limit = 5 # Reduced history limit
         for key in ['recent_orders', 'recent_allocations', 'demand_history', 'stockout_history', 'pending_releases']:
             if key in cleaned_obs and isinstance(cleaned_obs[key], list):
                 cleaned_obs[key] = cleaned_obs[key][-history_limit:]

         # Round floats in numerical dicts/lists to 1 decimal place
         for key, value in cleaned_obs.items():
            if isinstance(value, dict):
                try:
                    cleaned_obs[key] = {k: round(float(v), 1) if isinstance(v, (float, int, np.number)) else v for k, v in value.items()}
                except (ValueError, TypeError): pass # Ignore if conversion fails
            elif isinstance(value, list):
                 try:
                     # Only round if elements are numeric
                     if all(isinstance(x, (float, int, np.number)) for x in value):
                         cleaned_obs[key] = [round(float(x), 1) for x in value]
                 except (ValueError, TypeError): pass

         # Specifically handle nested dicts like inventories, pipeline summaries etc.
         for key in ['inventories', 'warehouse_inventories', 'production_capacity',
                     'pipeline', 'inbound_pipeline', 'outbound_pipeline',
                     'downstream_inventory_summary', 'downstream_pipeline_summary',
                     'downstream_projected_demand_summary', 'epidemiological_data',
                     'projected_demand']: # Add more keys if needed
             if key in cleaned_obs and isinstance(cleaned_obs[key], dict):
                 nested_dict = cleaned_obs[key]
                 cleaned_nested = {}
                 for nk, nv in nested_dict.items():
                     if isinstance(nv, dict): # Handle 2 levels deep
                         try:
                             cleaned_nested[nk] = {nnk: round(float(nnv), 1) if isinstance(nnv, (float, int, np.number)) else nnv for nnk, nnv in nv.items()}
                         except (ValueError, TypeError): cleaned_nested[nk] = nv # Keep original if error
                     elif isinstance(nv, (float, int, np.number)):
                         try: cleaned_nested[nk] = round(float(nv), 1)
                         except (ValueError, TypeError): cleaned_nested[nk] = nv
                     else:
                         cleaned_nested[nk] = nv # Keep non-numeric as is
                 cleaned_obs[key] = cleaned_nested


         # Convert to JSON string, then potentially truncate if too long
         try:
             json_string = json.dumps(cleaned_obs, indent=None, separators=(',', ':'), default=str, sort_keys=True) # Sort keys for consistency
             if len(json_string) > max_len:
                  # Basic truncation
                  truncated_json = json_string[:max_len]
                  # Try to find the last comma or brace to make it potentially valid-ish
                  last_sep = max(truncated_json.rfind(','), truncated_json.rfind('}'), truncated_json.rfind(']'))
                  if last_sep > 0:
                      return truncated_json[:last_sep] + '...[TRUNCATED]"}'
                  else: # Fallback if no separator found in truncated part
                      return truncated_json + '...[TRUNCATED]"}'
             return json_string
         except TypeError as e:
             self._print(f"[yellow]Could not serialize observation to JSON for prompt: {e}. Using basic string representation.[/]")
             fallback_str = f'"day": {observation.get("day")}, "inventories": {observation.get("inventories")}, "...(error serializing)"'
             return "{" + fallback_str[:max_len] + "}"

    # --- STEP 4: Refine LLM Prompts ---
    def _create_decision_prompt(self, observation: Dict, decision_type: str) -> str:
        """Create a detailed prompt for the OpenAI LLM with enhanced guidance."""
        current_day = observation.get("day", "N/A")
        agent_name = self._get_agent_name()

        # Basic Role Description
        role_desc = f"You are the {agent_name} in a pandemic supply chain simulation."
        # ... (rest of role description based on agent_type) ...
        if self.agent_type == "manufacturer":
             role_desc += " Your tasks are production planning and allocating drugs to regional distributors."
        elif self.agent_type == "distributor":
             role_desc += f" Your tasks are ordering drugs from the manufacturer and allocating available drugs to the hospital in your Region {self.agent_id}."
        elif self.agent_type == "hospital":
             role_desc += f" Your task is ordering drugs from the distributor in Region {self.agent_id} to meet patient demand."


        # Specific Task for this decision (ensure consistent JSON format guidance)
        json_guidance = "Respond ONLY with a valid JSON object with string keys (drug IDs, or region IDs if nested) and numerical amounts."
        if decision_type == "production": task = f"Determine optimal production quantity for each drug. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "allocation" and self.agent_type == "manufacturer": task = f"Determine optimal allocation amounts of AVAILABLE drugs to each requesting regional distributor. {json_guidance} Example: {{\"0\": {{\"0\": 100.0, \"1\": 50.0}}, \"1\": {{...}} }}"
        elif decision_type == "allocation" and self.agent_type == "distributor": task = f"Determine optimal allocation amounts of AVAILABLE drugs to the hospital in Region {self.agent_id}. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        elif decision_type == "order" and self.agent_type == "distributor": task = f"Determine optimal order quantity for each drug to request from the manufacturer. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "order" and self.agent_type == "hospital": task = f"Determine optimal order quantity for each drug to request from the distributor. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        else: task = f"Determine the appropriate {decision_type} actions. {json_guidance}"


        # Key Considerations - with added emphasis from Step 4
        considerations = f"""
        Consider the following factors:
        - Current day ({current_day}) and overall pandemic timeline.
        - Current inventory levels (own stock, warehouse if applicable). Check available amounts carefully.
        - Inbound and outbound pipeline quantities (drugs in transit).
        - Epidemiological data: current_cases, case_trend, projected_demand. ***PRIORITIZE FUTURE DEMAND BASED ON projected_demand and case_trend.*** React proactively to accelerating trends (positive case_trend).
        - Drug characteristics (criticality, base demand factor). Prioritize critical drugs.
        - Active and predicted supply chain disruptions (manufacturing, transportation). Increase buffers if risk is high.
        - Recent orders/allocations history (use mainly for context, not primary driver if forecasts are available).
        """
        # Add agent/decision specific considerations with PROACTIVE guidance
        if self.agent_type == "manufacturer":
            if decision_type == "allocation":
                considerations += """
        - Manufacturer Allocation: ***PRIORITIZE DOWNSTREAM NEEDS. Base allocations on projected demand across regions (see downstream_projected_demand_summary, estimate regional share using epidemiological_data). Be MORE GENEROUS if downstream inventory (downstream_inventory_summary) is low relative to projected demand or if case_trend is strongly positive, especially for critical drugs. You have warehouse inventory incoming; don't be afraid to allocate a large portion of your current on-hand 'inventories' if needed downstream.*** Allocate ONLY AVAILABLE inventory from your 'inventories' field. Use fair allocation principles if supply is limited. Adhere to batching schedules (is_batch_day, days_to_next_batch).
                """
            elif decision_type == "production":
                 considerations += """
        - Manufacturer Production: ***PRODUCE PROACTIVELY. Base production heavily on the SUM of projected demand across all regions (downstream_projected_demand_summary) and anticipate future growth indicated by positive case_trend values. Build warehouse buffers ahead of surges, especially for critical drugs or before batch allocation days (is_batch_day, days_to_next_batch).*** Consider production capacity and active disruptions.
                 """
        elif self.agent_type == "distributor":
             if decision_type == "allocation":
                  considerations += f"""
        - Distributor Allocation: Allocate ONLY to your single hospital (Region {self.agent_id}) based on their likely need (use projected_demand in epidemiological_data). Allocate ONLY AVAILABLE 'inventories'.
                 """
             elif decision_type == "order":
                  considerations += """
        - Distributor Ordering: Aim for sufficient stock to cover lead time demand + safety stock. ***PRIORITIZE FUTURE DEMAND. Use hospital's projected_demand and the region's case_trend. If case_trend is strongly positive, order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively.*** Consider manufacturer lead times and potential transport disruptions.
                 """
        elif self.agent_type == "hospital":
             if decision_type == "order":
                  considerations += """
        - Hospital Ordering: Order to meet patient demand. ***PRIORITIZE FUTURE DEMAND. Use your projected_demand and case_trend. If case_trend is strongly positive, order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively.*** Consider distributor lead times and potential transport disruptions.
                 """

        # Final prompt instruction (existing, reinforced)
        final_instruction = "Based *only* on the information provided above, determine the decision. Respond ONLY with the valid JSON object specified in the task, without any additional text, explanations, or markdown formatting."

        # Format Observation Data Concisely
        observation_summary = self._clean_observation_for_prompt(observation)

        # Combine into the final prompt
        full_prompt = f"""
        {role_desc}

        Current Simulation Day: {current_day}

        Your Task: {task}

        Key Considerations:
        {considerations}

        Current Situation & Data (JSON format, possibly truncated):
        {observation_summary}

        {final_instruction}
        """
        # self._print(f"DEBUG: Generated Prompt:\n{full_prompt}") # Uncomment for debugging prompts
        return full_prompt

    # --- Tool execution methods ---
    # (Keep existing tool execution methods: _run_epidemic_forecast_tool, _run_disruption_prediction_tool, etc.)
    # These methods remain the same, but how their output is used by the agents might change based on the logic above.

    def _run_epidemic_forecast_tool(self, observation: Dict) -> List[float]:
        """Run epidemic forecasting tool."""
        tool_name = "Epidemic Forecast"
        tool_result = [] # Default empty result
        try:
             if self.agent_type == "manufacturer":
                 all_forecasts = []
                 if "epidemiological_data" in observation:
                     for region_id_str, data in observation["epidemiological_data"].items():
                          current_cases = data.get("current_cases", 0)
                          # Find history for this region in memory (might be less reliable)
                          # Let's try using the direct case_history if available at top level first
                          history = []
                          if "case_history" in observation and region_id_str in observation["case_history"]:
                              history = observation["case_history"][region_id_str][-self.memory_length:] # Assuming structure {"case_history": {"0": [...], "1": [...]}}
                          else: # Fallback to memory scraping
                              history = [mem.get("epidemiological_data", {}).get(region_id_str, {}).get("current_cases")
                                         for mem in self.memory if mem.get("epidemiological_data", {}).get(region_id_str)]
                              history = [h for h in history if h is not None][-self.memory_length:]

                          forecast = self.tools.epidemic_forecast_tool(current_cases, history)
                          all_forecasts.append(forecast)
                 if all_forecasts:
                      min_len = min(len(f) for f in all_forecasts if f)
                      if min_len > 0:
                           avg_forecast = [np.mean([f[i] for f in all_forecasts if len(f) > i]) for i in range(min_len)]
                           tool_result = avg_forecast
                      else: tool_result = []
                 else: tool_result = []
             elif "epidemiological_data" in observation: # Distributor or Hospital
                 region_id_str = str(self.agent_id)
                 current_cases = observation["epidemiological_data"].get("current_cases", 0)
                 # Use memory scraping as before for D/H
                 history = [mem.get("epidemiological_data", {}).get("current_cases")
                            for mem in self.memory if mem.get("region_id") == self.agent_id and "epidemiological_data" in mem]
                 history = [h for h in history if h is not None][-self.memory_length:]
                 tool_result = self.tools.epidemic_forecast_tool(current_cases, history)
             else:
                 tool_result = []

             if self.verbose:
                 print_res = [f"{x:.1f}" for x in tool_result[:7]]
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output (first 7d):[/] {print_res}{'...' if len(tool_result) > 7 else ''}")

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty list.[/]")
             tool_result = []
        return tool_result

    def _run_disruption_prediction_tool(self, observation: Dict) -> Dict:
        """Run disruption prediction tool."""
        tool_name = "Disruption Prediction"
        tool_result = {"manufacturing": {}, "transportation": {}}
        try:
             current_day = observation.get("day", 0)
             # Use the collected known_disruptions history from self.known_disruptions
             tool_result = self.tools.disruption_prediction_tool(self.known_disruptions, current_day)
             self.disruption_predictions = tool_result # Store the prediction

             if self.verbose:
                  non_zero_preds = {
                      "manufacturing": {k: f"{v:.2f}" for k, v in tool_result.get("manufacturing", {}).items() if v > 0.01},
                      "transportation": {k: f"{v:.2f}" for k, v in tool_result.get("transportation", {}).items() if v > 0.01},
                  }
                  self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output (Prob > 0.01):[/] {non_zero_preds if any(non_zero_preds.values()) else '{}'}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty dict.[/]")
             tool_result = {"manufacturing": {}, "transportation": {}}
             self.disruption_predictions = tool_result # Store empty prediction on error
        return tool_result

    def _run_allocation_priority_tool(self, drug_info: Dict, region_requests: Dict, region_cases: Dict, available: float) -> Dict:
        """Run allocation priority tool."""
        tool_name = "Allocation Priority"
        tool_result = {}
        fallback_result = {}
        try:
            # Ensure keys are integers if needed by the tool
            int_key_requests = {int(k): float(v) for k, v in region_requests.items() if str(k).isdigit() and v is not None}
            int_key_cases = {int(k): float(v) for k, v in region_cases.items() if str(k).isdigit() and v is not None}

            # Ensure drug_info is passed correctly
            valid_drug_info = drug_info if isinstance(drug_info, dict) else {}

            tool_result = self.tools.allocation_priority_tool(valid_drug_info, int_key_requests, int_key_cases, max(0.0, float(available)))

            if self.verbose:
                print_res = {k: f"{v:.1f}" for k, v in tool_result.items()}
                self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")

        except Exception as e:
            self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
            # Fallback: Simple equal allocation among requesters
            num_requesters = len(region_requests)
            if num_requesters > 0 and available > 0:
                 # Filter only positive requests for fallback calculation
                 positive_requests = {r: req for r, req in region_requests.items() if req > 0}
                 num_positive_requesters = len(positive_requests)
                 if num_positive_requesters > 0:
                      equal_share = available / num_positive_requesters
                      fallback_result = {r: min(req, equal_share) for r, req in positive_requests.items()}
                 else: fallback_result = {}
            else: fallback_result = {}

            if self.verbose:
                 print_res = {k: f"{v:.1f}" for k, v in fallback_result.items()}
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using fallback allocation:[/]{print_res}")
            tool_result = fallback_result

        return tool_result


    def _run_optimal_order_quantity_tool(self, inventory: float, pipeline: float, demand_forecast: List[float], lead_time: int = 3, criticality: float = 1.0) -> float:
         """Run optimal order quantity tool."""
         tool_name = "Optimal Order Quantity"
         tool_result = 0.0
         try:
             # Adjust safety stock factor based on criticality
             safety_stock_factor = 1.0 + ((criticality - 1) / 3.0) * 1.0 # Scales 1.0 to 2.0
             safety_stock_factor = max(1.0, safety_stock_factor)

             # Convert forecast elements to float, handling potential None or non-numeric types
             cleaned_forecast = [float(d) for d in demand_forecast if isinstance(d, (int, float, np.number))]

             # Call the tool (which now handles trend internally)
             tool_result = self.tools.optimal_order_quantity_tool(
                 float(inventory), float(pipeline), cleaned_forecast, int(lead_time), float(safety_stock_factor)
                 )

             if self.verbose:
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {tool_result:.1f}")
         except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning 0.0[/]")
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
            # Ensure history is a list
            stockout_history = stockout_history if isinstance(stockout_history, list) else []
            tool_result = self.tools.criticality_assessment_tool(
                drug_info, stockout_history, float(unfulfilled), max(1.0, float(total)) # Ensure floats, avoid DivByZero
            )

            if self.verbose:
                print_res = f"Score: {tool_result.get('criticality_score', 'N/A')}, Category: {tool_result.get('category', 'N/A')}"
                self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] {print_res}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] [red]Error - Using fallback assessment.[/]")
             tool_result = fallback_assessment
        return tool_result


# --- END OF FILE src/agents/base.py ---