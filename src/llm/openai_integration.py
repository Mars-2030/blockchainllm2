"""
OpenAI API integration for the pandemic supply chain simulation.
"""

import json
import time
from typing import List, Dict, Optional
import openai


class OpenAILLMIntegration:
    """
    Integration class for OpenAI's API with the Pandemic Supply Chain simulation.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo", 
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        console=None
    ):
        """
        Initialize the OpenAI API integration.

        Args:
            api_key: OpenAI API key
            model_name: OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o")
            temperature: Generation temperature
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries in seconds
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.console = console

        # Initialize the OpenAI client
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            if self.console: # Check if console was passed
                self.console.print(f"[green]OpenAI client initialized successfully for model: {self.model_name}[/]")
        except Exception as e:
            if self.console: # Check if console was passed
                self.console.print(f"[bold red]Error initializing OpenAI client: {e}[/]")
            raise # Re-raise the exception
    
    def _print(self, message):
        """Helper to safely print using the stored console."""
        if self.console:
            self.console.print(message)

    def _make_api_call(self, messages: List[Dict], expect_json: bool = False) -> Optional[str]:
        """ Helper function to make API calls with retries """
        api_params = {
             "model": self.model_name,
             "messages": messages,
             "temperature": self.temperature,
             "max_tokens": 1024, # Adjust as needed
             "top_p": 0.95, # Optional
             # Add response_format for JSON if requested and model supports it
        }
        # Check if the model likely supports JSON mode (simple check, enhance if needed)
        # gpt-3.5-turbo-1106 and later, gpt-4-1106-preview and later, gpt-4o support JSON mode
        if expect_json and ("gpt-4" in self.model_name or "gpt-3.5-turbo-1106" in self.model_name or "gpt-3.5-turbo-0125" in self.model_name):
             api_params["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                return content.strip() if content else None
            except openai.RateLimitError as e:
                wait_time = self.retry_delay * (2 ** attempt) # Exponential backoff
                self._print(f"[yellow]OpenAI RateLimitError: {e}. Retrying in {wait_time:.1f}s...[/]")
                time.sleep(wait_time)
            except openai.APIError as e:
                self._print(f"[yellow]OpenAI APIError: {e}. Retrying in {self.retry_delay:.1f}s... (Attempt {attempt+1}/{self.max_retries})[/]")
                time.sleep(self.retry_delay)
            except Exception as e:
                self._print(f"[red]OpenAI call failed unexpectedly: {e} (Attempt {attempt+1}/{self.max_retries})[/]")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self._print(f"[bold red]OpenAI call failed after {self.max_retries} attempts.[/]")
                    return None
        return None # Failed after all retries


    def generate_reasoning(self, prompt: str) -> str:
        """Generate reasoning using OpenAI API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant providing reasoning for supply chain decisions."},
            {"role": "user", "content": prompt}
        ]
        reasoning = self._make_api_call(messages, expect_json=False)

        if reasoning:
            return reasoning
        else:
            return "ERROR: OpenAI API call for reasoning failed after multiple attempts. Using fallback reasoning."


    def generate_structured_decision(self, prompt: str, decision_type: str) -> Dict:
        """Generate a structured decision using OpenAI API, aiming for JSON output."""
        system_prompt = f"""
        You are a pandemic supply chain decision agent. Based on the user's input, make a {decision_type} decision.
        IMPORTANT: Respond ONLY with a valid JSON object representing the decision. Do NOT include any explanations, comments, markdown code blocks, or introductory text outside the JSON structure itself.
        The JSON keys should be appropriate for the decision type (e.g., drug IDs as strings, region IDs as strings if nested). Values should be numerical amounts.
        Example for manufacturer allocation: {{"0": {{"0": 100.0, "1": 50.0}}, "1": {{...}} }}
        Example for production/order/distributor allocation: {{"0": 1000.0, "1": 500.0}}
        Ensure all keys and string values are enclosed in double quotes. Ensure the final output is a single, valid JSON object starting with {{ and ending with }}.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response_text = self._make_api_call(messages, expect_json=True) # Request JSON format

        if not response_text:
            self._print(f"[red]Failed to get response from OpenAI for structured decision ({decision_type}).[/]")
            return {}

        # Try parsing the response as JSON
        parsed_json = {}
        try:
            # Sometimes the model might still wrap the JSON in backticks or explanations
            # Find the first '{' and the last '}'
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_text[json_start : json_end + 1]
                parsed_json = json.loads(json_str)

            else: # If no braces found, maybe it's just the JSON?
                try:
                    parsed_json = json.loads(response_text)
                except json.JSONDecodeError:
                     self._print(f"[red]Could not find valid JSON structure in OpenAI response:\n{response_text}[/]")
                     return {}

            # Basic validation: is it a dictionary?
            if isinstance(parsed_json, dict):
                 # Convert numerical values from potential strings if needed (robustness)
                 # This part depends heavily on expected JSON structure. Example:
                 cleaned_json = {}
                 for k, v in parsed_json.items():
                      try:
                           # Try converting key if it should be int (e.g., drug_id)
                           # Keep keys as strings as returned by API, convert later if needed
                           key_str = str(k)
                      except ValueError:
                           key_str = k # Keep as string if not int

                      if isinstance(v, dict): # Nested structure (e.g., allocation)
                           cleaned_sub_dict = {}
                           for sk, sv in v.items():
                                try:
                                     # Try converting subkey if it should be int (e.g., region_id)
                                     # Keep subkeys as strings
                                     subkey_str = str(sk)
                                except ValueError:
                                     subkey_str = sk
                                try:
                                     # Convert value to float
                                     cleaned_sub_dict[subkey_str] = float(sv)
                                except (ValueError, TypeError):
                                     # If conversion fails, keep original or set to 0? Let's log and keep 0.
                                     self._print(f"[yellow]Failed to convert nested value '{sv}' to float for key {key_str}, subkey {subkey_str}. Setting to 0.[/]")
                                     cleaned_sub_dict[subkey_str] = 0.0
                           cleaned_json[key_str] = cleaned_sub_dict
                      else: # Simple key-value (e.g., production, order)
                           try:
                                # Convert value to float
                                cleaned_json[key_str] = float(v)
                           except (ValueError, TypeError):
                                # If conversion fails, keep original or set to 0? Let's log and keep 0.
                                self._print(f"[yellow]Failed to convert value '{v}' to float for key {key_str}. Setting to 0.[/]")
                                cleaned_json[key_str] = 0.0

                 return cleaned_json
            else:
                 self._print(f"[red]Parsed JSON is not a dictionary: {type(parsed_json)}[/]")
                 return {}

        except json.JSONDecodeError as e:
            self._print(f"[red]Failed to decode JSON response from OpenAI: {e}\nResponse was:\n{response_text}[/]")
            return {}
        except Exception as e:
             self._print(f"[red]Error processing/cleaning OpenAI JSON response: {e}[/]")
             return {}