
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
        self.disruption_predictions = {}

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)
            
    def add_to_memory(self, observation: Dict):
        """Add current observation to memory."""
        # Simple memory: list of past observations
        self.memory.append(observation)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
        # Update known disruptions based on current observation
        if "disruptions" in observation:
             current_disruptions_keys = {(d['type'], d.get('drug_id', d.get('region_id')), d['start_day'])
                                    for d in observation['disruptions']}
             # Keep known disruptions that are still active or started recently
             cutoff_day = observation.get('day', 0) - 30 # Keep history for 30 days? Adjust as needed
             self.known_disruptions = [d for d in self.known_disruptions
                                      if d.get('end_day', -1) >= observation.get('day', 0) or d.get('start_day', -1) >= cutoff_day]
             # Add new unique disruptions from current observation
             existing_keys = {(d['type'], d.get('drug_id', d.get('region_id')), d.get('start_day'))
                              for d in self.known_disruptions}
             for d in observation['disruptions']:
                  key = (d['type'], d.get('drug_id', d.get('region_id')), d['start_day'])
                  if key not in existing_keys:
                       self.known_disruptions.append(d)

    def decide(self, observation: Dict) -> Dict:
        """Make a decision (placeholder, override in subclasses)."""
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")
        # Add observation to memory before making decision
        self.add_to_memory(observation)
        # Subclasses will implement the actual decision logic
        return {} # Return empty dict from base class

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

    def _clean_observation_for_prompt(self, observation: Dict, max_len: int = 3000) -> str:
         """Creates a string representation of the observation, trying to keep it concise."""
         # Convert complex objects to simpler representations
         cleaned_obs = observation.copy()

         # Simplify drug info - keep name, criticality, base_demand
         if 'drug_info' in cleaned_obs:
             cleaned_obs['drug_info'] = {k: {'name': v.get('name'), 'criticality': v.get('criticality'), 'base_demand': v.get('base_demand')}
                                         for k, v in cleaned_obs['drug_info'].items()}

         # Simplify history fields - show only last N items or summary stats
         history_limit = 7 # Keep last 7 items for history fields
         for key in ['recent_orders', 'recent_allocations', 'demand_history', 'stockout_history', 'pending_releases']:
             if key in cleaned_obs and isinstance(cleaned_obs[key], list):
                 cleaned_obs[key] = cleaned_obs[key][-history_limit:] # Keep last N items

         # Convert to JSON string, then potentially truncate if too long
         try:
             # Use compact separators to save space
             json_string = json.dumps(cleaned_obs, indent=None, separators=(',', ':'), default=str) # Use default=str for non-serializable
             if len(json_string) > max_len:
                  # Basic truncation (could be smarter, maybe prioritize certain keys)
                  return json_string[:max_len] + '...[TRUNCATED]"}' # Try to keep valid JSON structure
             return json_string
         except TypeError as e:
             # Fallback if JSON conversion fails completely
             self._print(f"[yellow]Could not serialize observation to JSON for prompt: {e}. Using basic string representation.[/]")
             # Simple string representation as fallback
             fallback_str = f'"day": {observation.get("day")}, "inventories": {observation.get("inventories")}, "...(error serializing)"'
             return "{" + fallback_str[:max_len] + "}"

    def _create_decision_prompt(self, observation: Dict, decision_type: str) -> str:
        """Create a detailed prompt for the OpenAI LLM."""
        current_day = observation.get("day", "N/A")
        agent_name = self._get_agent_name()

        # Basic Role Description
        role_desc = f"You are the {agent_name} in a pandemic supply chain simulation."
        if self.agent_type == "manufacturer":
             role_desc += " Your tasks are production planning and allocating drugs to regional distributors."
        elif self.agent_type == "distributor":
             role_desc += f" Your tasks are ordering drugs from the manufacturer and allocating available drugs to the hospital in your Region {self.agent_id}."
        elif self.agent_type == "hospital":
             role_desc += f" Your task is ordering drugs from the distributor in Region {self.agent_id} to meet patient demand."

        # Specific Task for this decision
        if decision_type == "production": task = "Determine optimal production quantity for each drug (return JSON: {\"drug_id_str\": amount})."
        elif decision_type == "allocation" and self.agent_type == "manufacturer": task = "Determine optimal allocation amounts of available drugs to each requesting regional distributor (return JSON: {\"drug_id_str\": {\"region_id_str\": amount}})."
        elif decision_type == "allocation" and self.agent_type == "distributor": task = f"Determine optimal allocation amounts of available drugs to the hospital in Region {self.agent_id} (return JSON: {{\"drug_id_str\": amount}})."
        elif decision_type == "order" and self.agent_type == "distributor": task = "Determine optimal order quantity for each drug to request from the manufacturer (return JSON: {\"drug_id_str\": amount})."
        elif decision_type == "order" and self.agent_type == "hospital": task = "Determine optimal order quantity for each drug to request from the distributor (return JSON: {\"drug_id_str\": amount})."
        else: task = f"Determine the appropriate {decision_type} actions."

        # Key Considerations
        considerations = f"""
        Consider the following factors:
        - Current day ({current_day}) and overall pandemic timeline.
        - Current inventory levels (own stock, and warehouse if applicable). Check available amounts carefully before deciding.
        - Inbound and outbound pipeline quantities (drugs in transit).
        - Epidemiological data (current cases, trends, projected demand, forecasts) to anticipate needs. React proactively to accelerating trends.
        - Drug characteristics (criticality, base demand). Higher criticality drugs need higher priority and safety stock.
        - Active and predicted supply chain disruptions (manufacturing, transportation). Increase buffers if risk is high.
        - Recent orders received or allocations received/made. Review history.
        - For allocation: Regional needs (cases, requests). Prioritize critical drugs and high-need regions. Use fair allocation if supply is limited. Only allocate AVAILABLE inventory.
        - For Distributor allocation: Allocate ONLY to your single hospital (Region {self.agent_id}).
        - For ordering: Aim for sufficient stock to cover lead time demand + safety stock, considering forecasts and disruptions. Use optimal ordering principles. Be more aggressive if forecasts show strong growth.
        - For production: Meet anticipated demand, build safety stock, account for production capacity and disruptions. Consider warehouse levels. Increase production proactively if forecasts show strong growth.
        - Adhere to batching schedules for manufacturer allocations if applicable (check is_batch_day, days_to_next_batch). Produce more ahead of batch days.
        - Return ONLY the JSON decision object as specified in the task description. Do not include explanations or any other text.
        """

        # Format Observation Data Concisely
        # Using helper to serialize and potentially truncate
        observation_summary = self._clean_observation_for_prompt(observation)

        # Combine into the final prompt for the user role
        full_prompt = f"""
        {role_desc}

        Current Simulation Day: {current_day}

        Your Task: {task}

        Key Considerations:
        {considerations}

        Current Situation & Data (JSON format, possibly truncated):
        {observation_summary}

        Based *only* on the information provided above, determine the {decision_type} decision. Respond only with the valid JSON object.
        """
        return full_prompt

    def _simulate_llm_reasoning(self, prompt: str) -> str:
        """Use OpenAI API for reasoning."""
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]🧠 {agent_name} reasoning with OpenAI API...[/]")

        reasoning = self.openai.generate_reasoning(prompt)

        if self.verbose:
             # Limit printing reasoning length in verbose mode
             max_len = 300
             display_reasoning = reasoning[:max_len] + "..." if len(reasoning) > max_len else reasoning
             self._print(f"[{Colors.REASONING}]📝 Reasoning: {display_reasoning}[/]")
        return reasoning

    # --- Tool execution methods ---
    def _run_epidemic_forecast_tool(self, observation: Dict) -> List[float]:
        """Run epidemic forecasting tool."""
        tool_name = "Epidemic Forecast"
        tool_result = [] # Default empty result
        try:
             # Manufacturer needs aggregate or regional data
             if self.agent_type == "manufacturer":
                 # Simple avg forecast across regions for manufacturer context
                 all_forecasts = []
                 if "epidemiological_data" in observation:
                     for region_id_str, data in observation["epidemiological_data"].items():
                          current_cases = data.get("current_cases", 0)
                          # Find history for this region in memory
                          history = [mem.get("epidemiological_data", {}).get(region_id_str, {}).get("current_cases")
                                     for mem in self.memory if mem.get("epidemiological_data", {}).get(region_id_str)]
                          history = [h for h in history if h is not None][-self.memory_length:] # Filter None, limit history
                          forecast = self.tools.epidemic_forecast_tool(current_cases, history)
                          all_forecasts.append(forecast)
                 # Average the forecasts element-wise
                 if all_forecasts:
                      min_len = min(len(f) for f in all_forecasts if f) # Avoid error if a forecast is empty
                      if min_len > 0:
                           avg_forecast = [np.mean([f[i] for f in all_forecasts if len(f) > i]) for i in range(min_len)]
                           tool_result = avg_forecast # Store result
                      else: tool_result = []
                 else: tool_result = []
             # Distributor and Hospital use their specific region's data
             elif "epidemiological_data" in observation:
                 region_id_str = str(self.agent_id) # Agent ID is the region ID for D/H
                 current_cases = observation["epidemiological_data"].get("current_cases", 0)
                 # Find history for this agent's region in memory
                 history = [mem.get("epidemiological_data", {}).get("current_cases")
                            for mem in self.memory if mem.get("region_id") == self.agent_id and "epidemiological_data" in mem] # Match region_id and ensure key exists
                 history = [h for h in history if h is not None][-self.memory_length:] # Filter None, limit history
                 tool_result = self.tools.epidemic_forecast_tool(current_cases, history) # Store result
             else:
                 tool_result = [] # No data available

             # --- Add verbose printing ---
             if self.verbose:
                 # Limit forecast print length
                 print_res = [f"{x:.1f}" for x in tool_result[:7]] # Show first 7 days forecast
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output (first 7d):[/] {print_res}{'...' if len(tool_result) > 7 else ''}")
             # --- End Add ---

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             # --- Add verbose printing on error ---
             if self.verbose:
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty list.[/]")
             # --- End Add ---
             tool_result = [] # Ensure result is empty list on error

        return tool_result # Return stored result

    def _run_disruption_prediction_tool(self, observation: Dict) -> Dict:
        """Run disruption prediction tool."""
        tool_name = "Disruption Prediction"
        tool_result = {"manufacturing": {}, "transportation": {}} # Default empty result
        try:
             # Use the collected known_disruptions history
             current_day = observation.get("day", 0)
             # Need to format known_disruptions correctly if tool expects specific dict keys
             formatted_history = self.known_disruptions # Assuming format is compatible
             tool_result = self.tools.disruption_prediction_tool(formatted_history, current_day) # Store result
             # --- Add verbose printing ---
             if self.verbose:
                  # Only print non-zero predictions for brevity
                  non_zero_preds = {
                      "manufacturing": {k: f"{v:.2f}" for k, v in tool_result.get("manufacturing", {}).items() if v > 0.01},
                      "transportation": {k: f"{v:.2f}" for k, v in tool_result.get("transportation", {}).items() if v > 0.01},
                  }
                  # Print even if empty to show tool ran
                  self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output (Prob > 0.01):[/] {non_zero_preds if any(non_zero_preds.values()) else '{}'}")
             # --- End Add ---

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             # --- Add verbose printing on error ---
             if self.verbose:
                  self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty dict.[/]")
             # --- End Add ---
             tool_result = {"manufacturing": {}, "transportation": {}} # Ensure result is empty dict on error

        return tool_result # Return stored result

    def _run_allocation_priority_tool(self, drug_info: Dict, region_requests: Dict, region_cases: Dict, available: float) -> Dict:
        """Run allocation priority tool."""
        tool_name = "Allocation Priority"
        tool_result = {} # Default empty result
        fallback_result = {}
        try:
            # Ensure region_requests keys are integers if needed by the tool
            int_key_requests = {int(k): v for k, v in region_requests.items() if isinstance(k, (int, str)) and str(k).isdigit()}
            int_key_cases = {int(k): v for k, v in region_cases.items() if isinstance(k, (int, str)) and str(k).isdigit()}
            tool_result = self.tools.allocation_priority_tool(drug_info, int_key_requests, int_key_cases, available) # Store result
            # --- Add verbose printing ---
            if self.verbose:
                print_res = {k: f"{v:.1f}" for k, v in tool_result.items()}
                self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")
            # --- End Add ---

        except Exception as e:
            self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
            # Fallback: Simple equal allocation among requesters
            num_requesters = len(region_requests)
            if num_requesters > 0 and available > 0:
                 equal_share = available / num_requesters
                 fallback_result = {r: min(req, equal_share) for r, req in region_requests.items() if req > 0} # Only allocate to positive requests
            else: fallback_result = {}
             # --- Add verbose printing on error ---
            if self.verbose:
                 print_res = {k: f"{v:.1f}" for k, v in fallback_result.items()}
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using fallback allocation:[/]{print_res}")
            # --- End Add ---
            tool_result = fallback_result # Assign fallback to result

        return tool_result # Return stored result or fallback

    def _run_optimal_order_quantity_tool(self, inventory: float, pipeline: float, demand_forecast: List[float], lead_time: int = 3, criticality: float = 1.0) -> float:
         """Run optimal order quantity tool."""
         tool_name = "Optimal Order Quantity"
         tool_result = 0.0 # Default result
         try:
             # Adjust safety stock factor based on criticality (code15 logic: 1.0 to 2.0 range approx)
             # Example: Base 1.0, add up to 1.0 based on criticality (value 1 to 4)
             safety_stock_factor = 1.0 + ((criticality - 1) / 3.0) * 1.0  # Scales from 1.0 (low) to 2.0 (critical)
             safety_stock_factor = max(1.0, safety_stock_factor) # Ensure minimum factor of 1

             tool_result = self.tools.optimal_order_quantity_tool(inventory, pipeline, demand_forecast, lead_time, safety_stock_factor) # Store result
             # --- Add verbose printing ---
             if self.verbose:
                 self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {tool_result:.1f}")
             # --- End Add ---

         except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             # --- Add verbose printing on error ---
             if self.verbose:
                  self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning 0.0[/]")
             # --- End Add ---
             tool_result = 0.0 # Ensure result is 0.0 on error

         return tool_result # Return stored result or 0.0

    def _run_criticality_assessment_tool(self, drug_info: Dict, stockout_history: List[Dict], unfulfilled: float, total: float) -> Dict:
        """Run criticality assessment tool."""
        tool_name = "Criticality Assessment"
        fallback_assessment = {
                 "drug_name": drug_info.get("name", "Unknown"),
                 "criticality_score": 0, "category": "Error",
                 "stockout_days_recent": len(stockout_history),
                 "unfulfilled_percentage_recent": 0,
                 "recommendations": ["Assessment tool failed."]
             }
        tool_result = fallback_assessment # Default result
        try:
            tool_result = self.tools.criticality_assessment_tool(drug_info, stockout_history, unfulfilled, max(1, total)) # Store result, avoid DivByZero
            # --- Add verbose printing ---
            if self.verbose:
                # Print only key fields for brevity
                print_res = f"Score: {tool_result.get('criticality_score', 'N/A')}, Category: {tool_result.get('category', 'N/A')}"
                self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] {print_res}")
            # --- End Add ---

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             # --- Add verbose printing on error ---
             if self.verbose:
                  self._print(f"[{Colors.YELLOW}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] [red]Error - Using fallback assessment.[/]")
             # --- End Add ---
             tool_result = fallback_assessment # Assign fallback

        return tool_result # Return stored result or fallback

