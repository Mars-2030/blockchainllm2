# --- START OF FILE config.py ---

"""
Configuration settings and utilities for the pandemic supply chain simulation.
"""

import os
import numpy as np
from rich.console import Console
from rich.terminal_theme import MONOKAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Create rich console
console = Console(record=True, width=120)

# Set random seed for reproducibility
np.random.seed(42)

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Blockchain Configuration ---
NODE_URL = os.getenv("NODE_URL", "http://127.0.0.1:8545") # Default to localhost
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
# Default location where deploy script saves ABI
CONTRACT_ABI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'blockchain', 'SupplyChainData.abi.json'))
# Account private key for sending transactions (owner role)
BLOCKCHAIN_PRIVATE_KEY = os.getenv("PRIVATE_KEY")


# Check OpenAI API key
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or OPENAI_API_KEY == "Insert the API key":
    console.print("[bold red]Error: Please set 'OPENAI_API_KEY' in the .env file.[/]")
    # Decide if simulation should exit or just disable LLM
    # exit(1)

# Check Blockchain config (only warn if needed for blockchain mode)
def check_blockchain_config():
    if not CONTRACT_ADDRESS:
        console.print("[bold yellow]Warning: CONTRACT_ADDRESS not found in .env. Blockchain features requiring address will fail.[/]")
        return False
    if not os.path.exists(CONTRACT_ABI_PATH):
         console.print(f"[bold yellow]Warning: Contract ABI not found at expected path: {CONTRACT_ABI_PATH}. Blockchain features will fail.[/]")
         return False
    if not BLOCKCHAIN_PRIVATE_KEY:
         console.print("[bold yellow]Warning: BLOCKCHAIN_PRIVATE_KEY not found in .env. Blockchain write operations (e.g., updating cases) will fail.[/]")
         # Allow read-only operation? For now, just warn.
    return True


class Colors:
    """Terminal color themes for the simulation."""
    MANUFACTURER = "blue"
    DISTRIBUTOR = "green"
    HOSPITAL = "magenta"
    YELLOW = "yellow"
    CYAN = "cyan"
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    MAGENTA = "magenta"
    BLACK = "black"
    BOLD = "bold"
    REASONING = "yellow"
    DECISION = "cyan"
    DIM = "dim" 
    STATE = "white"
    STOCKOUT = "red"
    IMPACT = "bold red"
    DAY_HEADER = "black on cyan bold"
    EPIDEMIC_STATE = "blue"
    TOOL_OUTPUT = "yellow"
    LLM_DECISION = "cyan"
    FALLBACK = "yellow bold"
    RULE = "magenta"
    BLOCKCHAIN = "bright_black" # Color for blockchain messages

    @staticmethod
    def styled_text(text, style):
        return f"[{style}]{text}[/]"

def save_console_html(console_obj: Console, filename="simulation_report_openai.html", output_folder="output"):
    """Save the console output to an HTML file."""
    output_path = os.path.join(output_folder, filename)
    try:
        console_obj.print(f"[dim]Attempting to export HTML to {output_path}...[/]") # Use passed console
        # Ensure console has recorded content before exporting
        if not console_obj.record:
             console_obj.print("[yellow]Console recording is disabled. Cannot export HTML.[/]")
             return

        html_content = console_obj.export_html(theme=MONOKAI)
        if not html_content or len(html_content) < 500:
             console_obj.print(f"[bold red]Warning: Exported HTML content seems empty or too short. Length: {len(html_content)}[/]")

        html_content = html_content.replace("<title>Rich", "<title>Pandemic Supply Chain Simulation Report (OpenAI + Blockchain)</title>")

        with open(output_path, "w", encoding="utf-8") as file:
            bytes_written = file.write(html_content)
        console_obj.print(f"[bold green]✓ Console output saved to HTML file: '{output_path}' ({bytes_written} bytes written)[/]")

    except Exception as e:
        console_obj.print(f"[bold red]Error saving console output to HTML: {e}[/]")
        # console_obj.print_exception(show_locals=False)

def ensure_folder_exists(console_obj: Console, folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        console_obj.print(f"[green]Created output folder: {folder_path}[/]")
    return folder_path

# --- END OF FILE config.py ---