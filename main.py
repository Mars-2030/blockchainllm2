#!/usr/bin/env python3
"""
Pandemic Supply Chain Simulation - Main Entry Point
"""

import argparse
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import console, Colors, ensure_folder_exists, save_console_html, OPENAI_API_KEY
from src.scenario.generator import PandemicScenarioGenerator
from src.scenario.visualizer import (
    visualize_epidemic_curves,
    visualize_drug_demand,
    visualize_disruptions,
    visualize_sir_components,  
    visualize_sir_simulation   
)
from src.environment.supply_chain import PandemicSupplyChainEnvironment
from src.environment.metrics import track_service_levels, visualize_service_levels, visualize_performance, visualize_inventory_levels
from src.tools import PandemicSupplyChainTools
from src.llm.openai_integration import OpenAILLMIntegration
from src.agents.manufacturer import create_openai_manufacturer_agent
from src.agents.distributor import create_openai_distributor_agent
from src.agents.hospital import create_openai_hospital_agent
from src.blockchain.interface import BlockchainInterface

import datetime


# Create rich console
# console = Console(record=True, width=120)

def run_openai_pandemic_simulation(
    console: Console, 
    openai_api_key: str,
    num_regions: int = 5,
    num_drugs: int = 3,
    simulation_days: int = 180,
    pandemic_severity: float = 0.8,
    disruption_probability: float = 0.1,
    warehouse_release_delay: int = 3,
    allocation_batch_frequency: int = 7,
    model_name: str = "gpt-3.5-turbo",
    visualize: bool = True,
    verbose: bool = True,
    use_colors: bool = True,
    output_folder: str = "output",
    blockchain_interface = None,
    use_blockchain: bool = False
):
    """Run simulation with OpenAI-powered agents."""

    if not use_colors: console.no_color = True
    console.print("[bold]Initializing OpenAI-powered pandemic supply chain simulation...[/]")

    # Create scenario and environment
    scenario_generator = PandemicScenarioGenerator(
        console=console, # Pass the console object
        num_regions=num_regions, num_drugs=num_drugs,
        scenario_length=simulation_days, pandemic_severity=pandemic_severity,
        disruption_probability=disruption_probability
    )
    environment = PandemicSupplyChainEnvironment(
        scenario_generator,
        blockchain_interface=blockchain_interface,
        use_blockchain=use_blockchain,
        console=console
    )
    environment.warehouse_release_delay = warehouse_release_delay
    environment.allocation_batch_frequency = allocation_batch_frequency
    environment.verbose = verbose

    # Create tools
    tools = PandemicSupplyChainTools()

    # Create OpenAI integration
    try:
        openai_integration = OpenAILLMIntegration(openai_api_key, model_name, console=console )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize OpenAI Integration: {e}. Aborting simulation.[/]")
        return None # Indicate failure

    # Create OpenAI-powered agents
    manufacturer = create_openai_manufacturer_agent(tools, openai_integration, verbose=verbose, console=console)
    distributors = [create_openai_distributor_agent(r, tools, openai_integration,num_regions=num_regions, verbose=verbose, console=console) for r in range(num_regions)]
    hospitals = [create_openai_hospital_agent(r, tools, openai_integration, verbose=verbose, console=console) for r in range(num_regions)]

    # Reset environment and metrics
    observations = environment.reset()
    metrics_history = {"stockouts": [], "unfulfilled_demand": [], "patient_impact": []}

    console.print(f"[bold]Running simulation for {simulation_days} days using {model_name}...[/]")
    start_time = time.time()

    # Simulation loop
    for day_index in range(simulation_days): 
        current_sim_day = day_index + 1 # Actual simulation day number (1-based)
        # if (day_index % 10 == 0 and not verbose) or verbose: # Print header periodically or always if verbose
        #     console.print(f"SIMULATION DAY {current_sim_day}/{simulation_days}", style="cyan")
        console.rule(f"[bold cyan] Starting Day {current_sim_day}/{simulation_days} [/bold cyan]", style="cyan")

                   # --- Print Daily Epidemic State (if verbose) ---
        if verbose:
            epi_table = Table(title=f"Epidemic State - Day {current_sim_day}", show_header=True, header_style="bold magenta", box=box.SIMPLE)
            epi_table.add_column("Region", style="cyan")
            epi_table.add_column("Active Cases", style="white", justify="right")
            epi_table.add_column("Trend (7d)", style="yellow", justify="right")

            scenario = environment.scenario # Get scenario object
            num_regions_in_scenario = len(scenario.regions) # Use actual num regions from scenario

            for r_id in range(num_regions_in_scenario):
                 region_name = scenario.regions[r_id].get("name", f"Region-{r_id+1}")
                 # Ensure we don't go out of bounds for epidemic_curves
                 if r_id in scenario.epidemic_curves:
                      curve = scenario.epidemic_curves[r_id]
                      current_idx = min(day_index, len(curve) - 1) # Use day_index (0-based) for curve access
                      prev_idx = max(0, current_idx - 7)

                      if current_idx >= 0 and len(curve) > current_idx:
                           current_cases = curve[current_idx]
                           # Ensure prev_idx is valid before accessing
                           prev_cases = curve[prev_idx] if len(curve) > prev_idx else 0
                           trend = current_cases - prev_cases if current_idx >= 7 else current_cases # Approx trend if less than 7 days history
                           epi_table.add_row(region_name, f"{current_cases:.0f}", f"{trend:+.0f}")
                      else:
                            epi_table.add_row(region_name, "[dim]N/A[/]", "[dim]N/A[/]")
                 else:
                       epi_table.add_row(region_name, "[dim]No Data[/]", "[dim]No Data[/]")

            console.print(epi_table)
            console.print()
        

        # --- Get Decisions ---
        all_actions = {}
        
        # Manufacturer
        try:
            manu_decision = manufacturer.decide(observations["manufacturer"])
            all_actions.update(manu_decision)
            # Verbose printing for manufacturer decisions
            if verbose and manu_decision.get("manufacturer_production"):
                prod_str = {k: f"{v:.1f}" for k, v in manu_decision["manufacturer_production"].items()}
                console.print(f"[blue]Manu Production:[/]{prod_str}")
            if verbose and manu_decision.get("manufacturer_allocation"):
                alloc_str = {k: {k2: f"{v2:.1f}" for k2, v2 in v.items()} for k, v in manu_decision["manufacturer_allocation"].items()}
                console.print(f"[blue]Manu Allocation:[/]{alloc_str}")
        except Exception as e:
            console.print(f"[bold red]Error during Manufacturer decision on day {current_sim_day}: {e}[/]")
            # Provide empty actions as fallback
            all_actions["manufacturer_production"] = {}
            all_actions["manufacturer_allocation"] = {}

        # Distributors
        dist_orders = {}
        dist_allocs = {}
        for dist_agent in distributors:
            try:
                dist_decision = dist_agent.decide(observations["distributors"][dist_agent.agent_id])
                if dist_decision.get("distributor_orders"):
                    dist_orders.update(dist_decision["distributor_orders"])
                if dist_decision.get("distributor_allocation"):
                    dist_allocs.update(dist_decision["distributor_allocation"])
                # Verbose printing
                if verbose and dist_decision.get("distributor_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_orders"].get(dist_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[green]Dist {dist_agent.agent_id} Orders:[/]{order_str}")
                if verbose and dist_decision.get("distributor_allocation"):
                    alloc_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_allocation"].get(dist_agent.agent_id, {}).items()}
                    if alloc_str: console.print(f"[green]Dist {dist_agent.agent_id} Allocation:[/]{alloc_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Distributor {dist_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                # Provide empty actions for this agent
                dist_orders[dist_agent.agent_id] = {}
                dist_allocs[dist_agent.agent_id] = {}

        all_actions["distributor_orders"] = dist_orders
        all_actions["distributor_allocation"] = dist_allocs

        # Hospitals
        hosp_orders = {}
        for hosp_agent in hospitals:
            try:
                hosp_decision = hosp_agent.decide(observations["hospitals"][hosp_agent.agent_id])
                if hosp_decision.get("hospital_orders"):
                    hosp_orders.update(hosp_decision["hospital_orders"])
                # Verbose printing
                if verbose and hosp_decision.get("hospital_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in hosp_decision["hospital_orders"].get(hosp_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[magenta]Hosp {hosp_agent.agent_id} Orders:[/]{order_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Hospital {hosp_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                # Provide empty actions for this agent
                hosp_orders[hosp_agent.agent_id] = {}

        all_actions["hospital_orders"] = hosp_orders

        # --- Step Environment ---
        try:
            observations, rewards, done, info = environment.step(all_actions)
        except Exception as e:
            console.print(f"[bold red]CRITICAL ERROR during environment step on day {current_sim_day}: {e}[/]")
            console.print_exception(show_locals=True) # Print stack trace
            console.print("Aborting simulation.")
            break # Exit simulation loop

        # --- Logging / Metrics ---
        if verbose:
            # Print summary state or specific warnings
            wh_inv = info.get('warehouse_inventory', {})
            manu_inv = info.get('manufacturer_inventory', {})
            console.print(f"Day {info['current_day']} End: WH Inv:{sum(wh_inv.values()):.0f}, Manu Inv:{sum(float(v) for v in manu_inv.values()):.0f}")
            # Check for significant stockouts
            day_stockouts = sum(s['unfulfilled'] for s in environment.stockout_history if s['day']==info['current_day']-1) # Stockout history is for day just ended
            if day_stockouts > 0:
                console.print(f"[yellow]Stockouts this day: {day_stockouts:.1f} units unfulfilled.[/]")

        # Record metrics history (ensure keys exist)
        metrics_history["stockouts"].append(info.get("stockouts", {}))
        metrics_history["unfulfilled_demand"].append(info.get("unfulfilled_demand", {}))
        metrics_history["patient_impact"].append(info.get("patient_impact", {}))

        if done:
            break

    # End simulation loop
    end_time = time.time()
    console.rule(f"\n[bold]Simulation complete. Total time: {end_time - start_time:.2f} seconds.[/]")


    # Final results calculation
    final_stockouts = environment.stockouts
    final_unfulfilled = environment.unfulfilled_demand
    final_impact = environment.patient_impact
    service_levels = track_service_levels(environment)

    results = {
        "total_stockouts": final_stockouts,
        "total_unfulfilled_demand": final_unfulfilled,
        "patient_impact": final_impact,
        "metrics_history": metrics_history,
        "service_levels": service_levels,
        "total_demand": environment.total_demand,
        "scenario_regions": scenario_generator.regions,
        "scenario_drugs": scenario_generator.drugs
    }

    # Generate visualizations if requested

    if visualize:
        console.print("[bold]Generating visualizations...[/]")
        try: 
            # Basic scenario visualizations
            visualize_epidemic_curves(scenario_generator, output_folder, console=console)
            visualize_drug_demand(scenario_generator, output_folder, console=console)
            visualize_disruptions(scenario_generator, output_folder, console=console)
            
            # SIR model specific visualizations
            visualize_sir_components(scenario_generator, output_folder, console=console)
            # Visualize SIR details for a few selected regions
            for region_id in range(min(3, num_regions)):  # First 3 regions or fewer
                visualize_sir_simulation(scenario_generator, region_id, output_folder, console=console)
            
            # Supply chain performance visualizations
            visualize_performance(environment, output_folder, console=console)
            visualize_inventory_levels(environment, output_folder, console=console)
            visualize_service_levels(environment, output_folder, console=console)
        except Exception as e:
            console.print(f"[red]Error during visualization: {e}[/]")
            console.print_exception(show_locals=True)  # More detailed error info

    return results

if __name__ == "__main__":
    # Use the API key provided earlier in the OPENAI_API_KEY variable.
    openai_key = OPENAI_API_KEY

    now = datetime.datetime.now()
    # Format it as YYYYMMDD_HHMMSS (or choose your preferred format)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Run pandemic supply chain simulation with OpenAI agents")
    parser.add_argument("--regions", type=int, default=5, help="Number of regions")
    parser.add_argument("--drugs", type=int, default=3, help="Number of drugs")
    parser.add_argument("--days", type=int, default=30, help="Simulation days") # Default to 30 for better results view
    parser.add_argument("--severity", type=float, default=0.8, help="Pandemic severity (0-1)")
    parser.add_argument("--disrupt-prob", type=float, default=0.1, help="Base disruption probability factor")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name (e.g., gpt-3.5-turbo, gpt-4-turbo-preview, gpt-4o)")
    parser.add_argument("--no-viz", action="store_false", dest="visualize", help="Disable visualizations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Less verbose output (API calls still shown)")
    parser.add_argument("--no-colors", action="store_false", dest="use_colors", help="Disable colored output")
    parser.add_argument("--folder", type=str, default="output", help="Folder for simulation output")
    parser.add_argument("--warehouse-delay", type=int, default=3, help="Warehouse release delay (days)")
    parser.add_argument("--allocation-batch", type=int, default=7, help="Allocation batch frequency (days, 1=daily)")
    parser.add_argument("--use-blockchain", action="store_true", default=False, help="Enable blockchain integration")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:8545", help="Ethereum node URL (e.g., Ganache, Infura)")

    args = parser.parse_args()

    if not args.use_colors: console.no_color = True

    output_folder_path = f"{args.folder}_{timestamp}_regions{args.regions}_drug{args.drugs}_days{args.days}"
    # Intro Panel
    console.print(Panel("[bold white]🦠 PANDEMIC SUPPLY CHAIN SIMULATION (using OpenAI) 🦠[/]", border_style="blue", expand=False, padding=(1,2)))

    # Config Table
    config_table = Table(title="Simulation Configuration", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Regions", str(args.regions))
    config_table.add_row("Drugs", str(args.drugs))
    config_table.add_row("Simulation Days", str(args.days))
    config_table.add_row("Pandemic Severity", f"{args.severity:.2f}")
    config_table.add_row("Disruption Probability Factor", f"{args.disrupt_prob:.2f}")
    config_table.add_row("Warehouse Delay", f"{args.warehouse_delay} days")
    config_table.add_row("Allocation Batch Frequency", f"{args.allocation_batch} days" if args.allocation_batch > 1 else "Daily")
    config_table.add_row("LLM Model", args.model)
    config_table.add_row("Visualizations", "Enabled" if args.visualize else "Disabled")
    config_table.add_row("Verbose Output", "Enabled" if args.verbose else "Disabled")
    config_table.add_row("Output Folder", output_folder_path)
    config_table.add_row("Blockchain", "Enabled" if args.use_blockchain else "Disabled")
    console.print(config_table)
    console.print() # Spacer

    # Ensure output folder exists
    output_folder = ensure_folder_exists(console,output_folder_path)

    # --- INITIALIZE BLOCKCHAIN INTERFACE ---
    blockchain_interface = None # Initialize as None
    use_blockchain_flag = False # Initialize flag

    if args.use_blockchain:
        console.print("\n[bold cyan]Attempting Blockchain Integration...[/]")
        try:
            # Pass the node URL if provided, otherwise it uses the default in the class
            blockchain_interface = BlockchainInterface(node_url=args.node_url)
            use_blockchain_flag = True # Set flag *only* on success
            console.print(f"[bold green]✓ Connected to Ethereum node at {blockchain_interface.node_url}[/]")
        except Exception as e:
            # CRITICAL FAILURE - HALT SIMULATION
            console.print(f"[bold red]❌ FATAL ERROR: Could not connect to blockchain or initialize contract interface: {e}[/]")
            console.print("[bold red]   Blockchain integration was requested (--use-blockchain) but failed.[/]")
            console.print("[bold red]   Halting simulation execution.[/]")
            # Save console output *before* exiting
            try:
                save_console_html(console, output_folder=output_folder, filename="simulation_error_report.html")
                console.print("[yellow]Partial error report saved to simulation_error_report.html[/]")
            except Exception as save_e:
                console.print(f"[red]Could not save error report: {save_e}[/]")
            exit(1) # Exit the program with a non-zero code indicating error
    else:
        # Blockchain explicitly disabled by argument
        use_blockchain_flag = False
        console.print("\n[yellow]Blockchain integration disabled by command-line argument.[/]")
        console.print("[yellow]Running in simulation-only mode.[/]")

    console.print("-" * 30) # Separator

    # --- Run Simulation ---
    results = run_openai_pandemic_simulation(
        console=console,
        openai_api_key=openai_key,
        num_regions=args.regions,
        num_drugs=args.drugs,
        simulation_days=args.days,
        pandemic_severity=args.severity,
        disruption_probability=args.disrupt_prob,
        warehouse_release_delay=args.warehouse_delay,
        allocation_batch_frequency=args.allocation_batch,
        model_name=args.model,
        visualize=args.visualize,
        verbose=args.verbose,
        use_colors=args.use_colors,
        output_folder=output_folder,
        blockchain_interface=blockchain_interface,
        use_blockchain=use_blockchain_flag
    )

    # --- Display Results ---
    if results: # Check if simulation ran successfully
        console.print(Panel("[bold white]Simulation Results Summary[/]", border_style="green", expand=False))

        # Get scenario info for names
        drug_names = {d['id']: d['name'] for d in results.get('scenario_drugs', [])}
        region_names = {r['id']: r['name'] for r in results.get('scenario_regions', [])}

        # Stockouts Summary
        console.print("\n[bold red]Total Stockout Days by Drug and Region:[/]")
        stockout_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        stockout_table_summary.add_column("Drug", style="cyan", min_width=10)
        stockout_table_summary.add_column("Region", style="magenta", min_width=10)
        stockout_table_summary.add_column("Stockout Days", style="white", justify="right", min_width=15)
        total_stockout_days = 0
        for drug_id, regions in results["total_stockouts"].items():
             drug_name = drug_names.get(drug_id, f"Drug {drug_id}")
             for region_id, count in regions.items():
                  region_name = region_names.get(region_id, f"Region {region_id}")
                  if count > 0: # Only show regions/drugs with stockouts
                       color = "red" if count > (args.days * 0.3) else "yellow" # Highlight if >30% days
                       stockout_table_summary.add_row(drug_name, region_name, f"[{color}]{count}[/]")
                       total_stockout_days += count
        if total_stockout_days == 0:
             console.print("[green]✓ No stockout days recorded across all regions and drugs.[/]")
        else:
             console.print(stockout_table_summary)
             stockout_severity_threshold = (args.days * args.regions * args.drugs) * 0.1 # e.g., 10% of possible stockout days
             color = "red" if total_stockout_days > stockout_severity_threshold * 2 else "yellow" if total_stockout_days > 0 else "green"
             console.print(f"  Total stockout days across system: [bold {color}]{total_stockout_days}[/]")

        # Unfulfilled Demand Summary
        total_unfulfilled = sum(sum(drug.values()) for drug in results["total_unfulfilled_demand"].values())
        # Try to get total demand for percentage calculation
        total_demand_all = sum(sum(drug.values()) for drug in results.get("total_demand", {}).values())
        percent_unfulfilled_str = f" ({ (total_unfulfilled / total_demand_all * 100) if total_demand_all > 0 else 0 :.1f}%)" if total_demand_all > 0 else ""
        color = "red" if total_unfulfilled > 10000 else "yellow" if total_unfulfilled > 0 else "green" # Adjusted threshold
        console.print(f"\n[bold]Total Unfulfilled Demand (units): [{color}]{total_unfulfilled:.1f}[/{color}]{percent_unfulfilled_str}")

        # Patient Impact Summary
        console.print("\n[bold red]Patient Impact Score by Region:[/]")
        impact_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        impact_table_summary.add_column("Region", style="magenta", min_width=10)
        impact_table_summary.add_column("Impact Score", style="white", justify="right", min_width=15)
        total_impact = sum(results["patient_impact"].values())
        for region_id, impact in results["patient_impact"].items():
             region_name = region_names.get(region_id, f"Region {region_id}")
             impact_color = "red" if impact > 10000 else "yellow" if impact > 100 else "green" # Adjusted threshold
             impact_table_summary.add_row(region_name, f"[{impact_color}]{impact:.1f}[/]")
        console.print(impact_table_summary)
        color = "red" if total_impact > 50000 else "yellow" if total_impact > 500 else "green" # Adjusted threshold
        console.print(f"  Total Patient Impact Score: [bold {color}]{total_impact:.1f}[/]")

        # Service Level Summary
        console.print("\n[bold cyan]Service Level Performance (% Demand Met):[/]")
        service_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        service_table_summary.add_column("Metric", style="cyan", min_width=15)
        service_table_summary.add_column("Service Level", style="white", justify="right", min_width=15)
        service_levels = results["service_levels"]
        if service_levels:
             avg_service = np.mean([item["service_level"] for item in service_levels]) if service_levels else 0
             min_service = min(item["service_level"] for item in service_levels) if service_levels else 0
             final_service = service_levels[-1]["service_level"] if service_levels else 0

             def get_service_color(level):
                  if level >= 95: return "green"
                  elif level >= 90: return "cyan"
                  elif level >= 80: return "yellow"
                  else: return "red"

             service_table_summary.add_row("Average", f"[{get_service_color(avg_service)}]{avg_service:.1f}%[/]")
             service_table_summary.add_row("Minimum", f"[{get_service_color(min_service)}]{min_service:.1f}%[/]")
             service_table_summary.add_row("Final Day", f"[{get_service_color(final_service)}]{final_service:.1f}%[/]")
             console.print(service_table_summary)
        else:
             console.print("[yellow]No service level data calculated.[/]")

        # Overall Performance Rating
        rating = "N/A"; rating_color="white"
        if service_levels: # Base rating on average service level
             avg_service = np.mean([item["service_level"] for item in service_levels])
             # Adjusted rating thresholds
             if avg_service >= 95 and total_impact < (50 * args.days): rating, rating_color = "Excellent", "green"
             elif avg_service >= 90 and total_impact < (200 * args.days): rating, rating_color = "Good", "cyan"
             elif avg_service >= 80 and total_impact < (1000 * args.days): rating, rating_color = "Fair", "yellow"
             else: rating, rating_color = "Poor", "red"
        console.print(f"\n[bold]Overall Supply Chain Performance:[/] [{rating_color}]{rating}[/]")

    else:
        console.print("[bold red]Simulation did not complete successfully. No results to display.[/]")

    # --- PRINT FINAL BLOCKCHAIN STATE ---
    if use_blockchain_flag and blockchain_interface:
        console.print("\n[bold cyan]Querying Final Blockchain State...[/]")
        try:
            blockchain_interface.print_contract_state()
        except Exception as e:
            console.print(f"[red]Error querying final blockchain state: {e}[/]")
    
    # Save console output to HTML
    save_console_html(console,output_folder=output_folder)
    console.print(f"\n[green]Visualizations and report saved to folder: '{output_folder}'[/]")