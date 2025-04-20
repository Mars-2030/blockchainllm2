"""
Configuration settings and utilities for the pandemic supply chain simulation.
"""

import os
import numpy as np
# from rich.console import Console
from rich.console import Console
from rich.terminal_theme import MONOKAI
from dotenv import load_dotenv 


# Load environment variables from .env file
load_dotenv()
# Create rich console
console = Console(record=True, width=120)

# Set random seed for reproducibility
np.random.seed(42)

# API key handling - Replace with your actual key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check API key
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or OPENAI_API_KEY == "Insert the API key":
    console.print("[bold red]Error: Please replace 'YOUR_OPENAI_API_KEY' in the code with the actual OpenAI API key.[/]")
    exit(1)

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
    STATE = "white"
    STOCKOUT = "red"
    IMPACT = "bold red"
    DAY_HEADER = "black on cyan bold"
    EPIDEMIC_STATE = "blue"
    TOOL_OUTPUT = "yellow"
    LLM_DECISION = "cyan"
    FALLBACK = "yellow bold"
    RULE = "magenta"

    @staticmethod
    def styled_text(text, style):
        return f"[{style}]{text}[/]"

    # @staticmethod
    # def disable_colors():
    #     console.no_color = True

def save_console_html(console: Console, filename="simulation_report_openai.html", output_folder="output"):
    """Save the console output to an HTML file."""
    output_path = os.path.join(output_folder, filename)
    try:
        console.print(f"[dim]Attempting to export HTML to {output_path}...[/]") # Add debug print
        html_content = console.export_html(theme=MONOKAI) # Use the passed console object
        if not html_content or len(html_content) < 500: # Check if content seems valid
             console.print(f"[bold red]Warning: Exported HTML content seems empty or too short. Length: {len(html_content)}[/]")

        html_content = html_content.replace("<title>Rich", "<title>Pandemic Supply Chain Simulation Report (OpenAI)</title>")

        with open(output_path, "w", encoding="utf-8") as file:
            bytes_written = file.write(html_content)
        # Use the passed console object for printing confirmation
        console.print(f"[bold green]✓ Console output saved to HTML file: '{output_path}' ({bytes_written} bytes written)[/]")

    except Exception as e:
        # Use the passed console object for printing errors
        console.print(f"[bold red]Error saving console output to HTML: {e}[/]")
        # console.print_exception(show_locals=False)

def ensure_folder_exists(console: Console, folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # Use the passed console object
        console.print(f"[green]Created output folder: {folder_path}[/]")
    return folder_path