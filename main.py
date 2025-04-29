# --- START OF FILE main.py ---

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
import os # Added for path joining
from typing import Optional, Dict # For type hinting

# Updated import order to match config first
from config import (
    console, Colors, ensure_folder_exists, save_console_html, OPENAI_API_KEY,
    NODE_URL, CONTRACT_ADDRESS, CONTRACT_ABI_PATH, BLOCKCHAIN_PRIVATE_KEY,
    check_blockchain_config
)
from src.scenario.generator import PandemicScenarioGenerator
from src.scenario.visualizer import (
    visualize_epidemic_curves,
    visualize_drug_demand,
    visualize_disruptions,
    visualize_sir_components,
    visualize_sir_simulation
)
from src.environment.supply_chain import PandemicSupplyChainEnvironment
from src.environment.metrics import track_service_levels, visualize_service_levels, visualize_performance, visualize_inventory_levels, visualize_blockchain_performance
from src.tools import PandemicSupplyChainTools # Import the class
from src.llm.openai_integration import OpenAILLMIntegration
from src.agents.manufacturer import create_openai_manufacturer_agent
from src.agents.distributor import create_openai_distributor_agent
from src.agents.hospital import create_openai_hospital_agent
# Import BlockchainInterface only if needed, handle potential import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Allows running without blockchain dependencies


import datetime

# Create rich console is now done in config.py

def run_pandemic_simulation(
    console: Console,
    openai_api_key: Optional[str],
    num_regions: int = 3, # Default back to 3 for common use
    num_drugs: int = 3,
    simulation_days: int = 180, # Increased default simulation length
    pandemic_severity: float = 0.8,
    disruption_probability: float = 0.1,
    # --- Use Consistent Default Warehouse Delay ---
    warehouse_release_delay: int = 1, # Default: 1 day delay
    # --------------------------------------------
    allocation_batch_frequency: int = 1, # Default: Daily allocation
    model_name: str = "gpt-3.5-turbo",
    visualize: bool = True,
    verbose: bool = True,
    use_colors: bool = True,
    output_folder: str = "output",
    blockchain_interface: Optional[BlockchainInterface] = None, # Allow None
    use_blockchain: bool = False,
    use_llm: bool = False
):
    """Run simulation with OpenAI-powered agents."""

    if not use_colors: console.no_color = True
    sim_mode = "LLM-Powered" if use_llm else "Rule-Based"
    sim_type = f"{sim_mode}" + (" + Blockchain" if use_blockchain else "")
    console.print(f"[bold]Initializing {sim_type} pandemic supply chain simulation...[/]")
    
    # Create scenario and environment
    scenario_generator = PandemicScenarioGenerator(
        console=console, # Pass the console object
        num_regions=num_regions, num_drugs=num_drugs,
        scenario_length=simulation_days, pandemic_severity=pandemic_severity,
        disruption_probability=disruption_probability
    )
    environment = PandemicSupplyChainEnvironment(
        scenario_generator,
        blockchain_interface=blockchain_interface, # Pass the interface object
        use_blockchain=use_blockchain,
        console=console
    )
    environment.warehouse_release_delay = warehouse_release_delay
    environment.allocation_batch_frequency = allocation_batch_frequency
    environment.verbose = verbose

    # Create tools instance
    tools = PandemicSupplyChainTools()

    # Conditional OpenAI Integration Initialization
    openai_integration = None
    if use_llm:
        if not openai_api_key:
            console.print("[bold red]Error: --use-llm flag requires OpenAI API key to be set in config or environment variables.[/]")
            return None # Indicate failure
        try:
            openai_integration = OpenAILLMIntegration(openai_api_key, model_name, console=console)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize OpenAI Integration: {e}. Aborting simulation.[/]")
            return None # Indicate failure
    else:
        console.print("[yellow]Running in Rule-Based mode. OpenAI Integration skipped.[/]")
        
    # Create agents, passing the blockchain interface instance if enabled
    manufacturer = create_openai_manufacturer_agent(
        tools=tools,
        openai_integration=openai_integration, # Will be None if use_llm is False
        num_regions=num_regions,
        memory_length=10,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface if use_blockchain else None,
        use_llm=use_llm # Pass the flag
    )
    distributors = [create_openai_distributor_agent(
        region_id=r,
        tools=tools,
        openai_integration=openai_integration, # Will be None if use_llm is False
        num_regions=num_regions,
        memory_length=10,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface if use_blockchain else None,
        use_llm=use_llm # Pass the flag
        ) for r in range(num_regions)]
    hospitals = [create_openai_hospital_agent(
        region_id=r,
        tools=tools,
        openai_integration=openai_integration, # Will be None if use_llm is False
        memory_length=10,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface if use_blockchain else None,
        use_llm=use_llm # Pass the flag
        ) for r in range(num_regions)]

    # Reset environment and metrics
    observations = environment.reset() # Reset should handle initializing blockchain state if needed
    metrics_history = {"stockouts": [], "unfulfilled_demand": [], "patient_impact": []}

    llm_model_info = f"using {model_name}" if use_llm else "using Rule-Based logic"
    console.print(f"[bold]Running simulation for {simulation_days} days {llm_model_info}...[/]")
    start_time = time.time()

    # Simulation loop
    for day_index in range(simulation_days):
        current_sim_day = day_index + 1 # Actual simulation day number (1-based)
        console.rule(f"[bold cyan] Starting Day {current_sim_day}/{simulation_days} [/bold cyan]", style="cyan")

        # --- (MODIFIED) UPDATE BLOCKCHAIN WITH CURRENT SIM CASES *BEFORE* DECISIONS ---
        if use_blockchain and blockchain_interface:
            # Get simulation cases for the START of the current day (index `day_index`)
            # The environment's current_day is still `day_index` before the step()
            sim_cases_today = environment.get_current_simulation_cases() # Uses environment.current_day which is day_index
            if verbose: console.print(f"[{Colors.BLOCKCHAIN}]Updating Blockchain cases for Day {current_sim_day} (Based on Sim State at start of day): {sim_cases_today}[/]")
            update_failed = False
            for region_id, cases_int in sim_cases_today.items():
                try:
                    # Make the update call
                    tx_result = blockchain_interface.update_regional_case_count(
                        region_id=int(region_id),
                        cases=cases_int
                    )
                    # Log failures but continue simulation
                    if tx_result is None or tx_result.get('status') != 'success':
                        console.print(f"[{Colors.BLOCKCHAIN}][yellow]BC Tx Failed (Update Cases R{region_id} Day {current_sim_day}): {tx_result.get('error', 'Unknown BC error') if tx_result else 'Comm error'}[/]")
                        update_failed = True
                except Exception as e:
                    console.print(f"[{Colors.BLOCKCHAIN}][red]BC Error calling update_case_data for R{region_id} Day {current_sim_day}: {e}[/]")
                    update_failed = True

            # Optional: Handle critical failures if needed
            # if update_failed:
            #     console.print(f"[bold red]CRITICAL: Blockchain update failed for one or more regions on Day {current_sim_day}. Simulation integrity potentially compromised.[/]")
            #     # Decide whether to continue or stop based on simulation requirements
            #     # break # Example: Stop simulation on critical failure
        # ----------------------------------------------------------------------

        # --- Print Daily Epidemic State (if verbose) ---
        # Table now shows Sim cases for the current day vs. BC cases (which should reflect the update just performed)
        if verbose:
            epi_table = Table(title=f"Epidemic State - Day {current_sim_day}", show_header=True, header_style="bold magenta", box=box.SIMPLE)
            epi_table.add_column("Region", style="cyan")
            epi_table.add_column("Cases (Sim)", style="white", justify="right") # Internal Simulation Cases for current day
            epi_table.add_column("Proj.Demand(Sum)", style="magenta", justify="right") # Projected Demand based on current day
            if use_blockchain and blockchain_interface:
                 epi_table.add_column("Cases (BC)", style=Colors.BLOCKCHAIN, justify="right") # Cases on BC (should now match Sim)

            scenario = environment.scenario # Get scenario object
            num_regions_in_scenario = len(scenario.regions) # Use actual num regions from scenario

            # Get simulation cases again for display consistency (should match what was sent to BC)
            sim_cases_display = environment.get_current_simulation_cases()

            for r_id in range(num_regions_in_scenario):
                 region_name = scenario.regions[r_id].get("name", f"Region-{r_id+1}")
                 bc_cases_str = "[dim]N/A[/]" # Default if BC disabled or error

                 sim_cases = sim_cases_display.get(r_id, 0)
                 sim_cases_str = f"{sim_cases:.0f}"

                 # Calculate projected demand using current day index
                 proj_demand_region = sum(scenario.get_daily_drug_demand(day=day_index, region_id=r_id, drug_id=d_id) for d_id in range(environment.num_drugs))
                 proj_demand_str = f"{proj_demand_region:.0f}"

                 # Query blockchain *after* the update attempt
                 if use_blockchain and blockchain_interface:
                     try:
                          # Note: This is a read operation for display, agents read separately via tool
                          bc_cases = blockchain_interface.get_regional_case_count(r_id)
                          bc_cases_str = f"{bc_cases}" if bc_cases is not None else "[red]Read Error[/]"
                     except Exception as e:
                          bc_cases_str = "[red]QueryErr[/]"

                 # Add row to table
                 if use_blockchain and blockchain_interface:
                       epi_table.add_row(region_name, sim_cases_str, proj_demand_str, bc_cases_str)
                 else:
                       epi_table.add_row(region_name, sim_cases_str, proj_demand_str)

            console.print(epi_table)
            console.print()


        # --- Get Decisions ---
        # Agents make decisions based on observations and potentially querying the *updated* blockchain state
        all_actions = {}
        manu_decision = {}; dist_orders = {}; dist_allocs = {}; hosp_orders = {}

        # Manufacturer
        try:
            manu_obs = observations.get("manufacturer", {}) # Observation now excludes direct cases/trend
            if manu_obs:
                manu_decision = manufacturer.decide(manu_obs) # Agent uses tool to get BC cases
            else:
                console.print("[yellow]Warning: No manufacturer observation found.[/]")
                manu_decision = {} # Empty decision if no observation
            all_actions.update(manu_decision or {}) # Use empty dict if decision failed

            # Verbose printing for manufacturer decisions
            if verbose and manu_decision.get("manufacturer_production"):
                prod_str = {k: f"{v:.1f}" for k, v in manu_decision["manufacturer_production"].items()}
                console.print(f"[{Colors.MANUFACTURER}]Manu Production:[/]{prod_str}")
            if verbose and manu_decision.get("manufacturer_allocation"):
                alloc_str = {k: {k2: f"{v2:.1f}" for k2, v2 in v.items()} for k, v in manu_decision["manufacturer_allocation"].items()}
                console.print(f"[{Colors.MANUFACTURER}]Manu Allocation Request:[/]{alloc_str}")
        except Exception as e:
            console.print(f"[bold red]Error during Manufacturer decision on day {current_sim_day}: {e}[/]")
            console.print_exception(show_locals=False)
            # Provide empty actions as fallback
            all_actions["manufacturer_production"] = {}
            all_actions["manufacturer_allocation"] = {}

        # Distributors
        for dist_agent in distributors:
            try:
                dist_obs = observations.get("distributors", {}).get(dist_agent.agent_id)
                if dist_obs:
                    dist_decision = dist_agent.decide(dist_obs)
                else:
                    console.print(f"[yellow]Warning: No observation found for Distributor {dist_agent.agent_id}.[/]")
                    dist_decision = {}

                if dist_decision.get("distributor_orders"):
                    dist_orders.update(dist_decision["distributor_orders"])
                if dist_decision.get("distributor_allocation"):
                    dist_allocs.update(dist_decision["distributor_allocation"])

                # Verbose printing
                if verbose and dist_decision.get("distributor_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_orders"].get(dist_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[{Colors.DISTRIBUTOR}]Dist {dist_agent.agent_id} Orders:[/]{order_str}")
                if verbose and dist_decision.get("distributor_allocation"):
                    alloc_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_allocation"].get(dist_agent.agent_id, {}).items()}
                    if alloc_str: console.print(f"[{Colors.DISTRIBUTOR}]Dist {dist_agent.agent_id} Allocation:[/]{alloc_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Distributor {dist_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                console.print_exception(show_locals=False)
                # Provide empty actions for this agent
                dist_orders[dist_agent.agent_id] = {}
                dist_allocs[dist_agent.agent_id] = {}
        all_actions["distributor_orders"] = dist_orders
        all_actions["distributor_allocation"] = dist_allocs

        # Hospitals
        for hosp_agent in hospitals:
            try:
                hosp_obs = observations.get("hospitals", {}).get(hosp_agent.agent_id)
                if hosp_obs:
                    hosp_decision = hosp_agent.decide(hosp_obs)
                else:
                    console.print(f"[yellow]Warning: No observation found for Hospital {hosp_agent.agent_id}.[/]")
                    hosp_decision = {}

                if hosp_decision.get("hospital_orders"):
                    hosp_orders.update(hosp_decision["hospital_orders"])
                # Verbose printing
                if verbose and hosp_decision.get("hospital_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in hosp_decision["hospital_orders"].get(hosp_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[{Colors.HOSPITAL}]Hosp {hosp_agent.agent_id} Orders:[/]{order_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Hospital {hosp_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                console.print_exception(show_locals=False)
                # Provide empty actions for this agent
                hosp_orders[hosp_agent.agent_id] = {}
        all_actions["hospital_orders"] = hosp_orders

        # --- Step Environment ---
        try:
            # Environment step now receives actions based on agent decisions
            # The _process_patient_demand method inside step() no longer updates the blockchain.
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
            pending_str = ", PendingAlloc: Yes" if info.get('pending_allocations') else ""
            # Manu Inv needs safe summing
            manu_inv_sum = sum(float(v) for v in manu_inv.values() if isinstance(v, (int, float)))
            console.print(f"Day {info['current_day']} End: WH Inv:{sum(wh_inv.values()):.0f}, Manu Inv:{manu_inv_sum:.0f}{pending_str}")
            # Check for significant stockouts
            # Stockout history index needs care, day just ended is current_sim_day
            day_stockouts_list = [s['unfulfilled'] for s in environment.stockout_history if isinstance(s, dict) and s.get('day') == day_index] # Use day_index (0-based) for history
            day_stockouts = sum(day_stockouts_list)
            if day_stockouts > 0:
                console.print(f"[{Colors.STOCKOUT}]Stockouts recorded for day {current_sim_day}: {day_stockouts:.1f} units unfulfilled.[/]")

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

    # Add scenario info to results for better reporting context
    results = {
        "total_stockouts": final_stockouts,
        "total_unfulfilled_demand": final_unfulfilled,
        "patient_impact": final_impact,
        "metrics_history": metrics_history,
        "service_levels": service_levels,
        "total_demand": environment.total_demand,
        "scenario_regions": scenario_generator.regions,
        "scenario_drugs": scenario_generator.drugs,
        "config_warehouse_delay": environment.warehouse_release_delay,
        "config_allocation_frequency": environment.allocation_batch_frequency
    }

    # Generate visualizations if requested
    if visualize:
        console.print("[bold]Generating visualizations...[/]")
        try:
            visualize_epidemic_curves(scenario_generator, output_folder, console=console)
            visualize_drug_demand(scenario_generator, output_folder, console=console)
            visualize_disruptions(scenario_generator, output_folder, console=console)
            # Visualize SIR components combined plot
            visualize_sir_components(scenario_generator, output_folder, console=console)
            # Visualize individual region SIR details (up to 3 regions)
            for region_id in range(min(3, num_regions)):
                 if scenario_generator.epidemic_curves.get(region_id) is not None: # Check if curve exists
                    visualize_sir_simulation(scenario_generator, region_id, output_folder, console=console)
                 else:
                    console.print(f"[yellow]Skipping SIR detail plot for Region {region_id}: No epidemic curve data.[/]")

            visualize_performance(environment, output_folder, console=console)
            visualize_inventory_levels(environment, output_folder, console=console)
            visualize_service_levels(environment, output_folder, console=console)
        except Exception as e:
            console.print(f"[red]Error during visualization: {e}[/]")
            console.print_exception(show_locals=True)

    return results

if __name__ == "__main__":
    openai_key = OPENAI_API_KEY
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Run pandemic supply chain simulation with OpenAI agents")
    parser.add_argument("--regions", type=int, default=3, help="Number of regions")
    parser.add_argument("--drugs", type=int, default=3, help="Number of drugs")
    parser.add_argument("--days", type=int, default=180, help="Simulation days") # Default increased
    parser.add_argument("--severity", type=float, default=0.8, help="Pandemic severity (0-1)")
    parser.add_argument("--disrupt-prob", type=float, default=0.1, help="Base disruption probability factor")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name (e.g., gpt-3.5-turbo, gpt-4-turbo-preview, gpt-4o)")
    parser.add_argument("--no-viz", action="store_false", dest="visualize", help="Disable visualizations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Less verbose output")
    parser.add_argument("--no-colors", action="store_false", dest="use_colors", help="Disable colored output")
    parser.add_argument("--folder", type=str, default="output", help="Base folder for simulation output")
    # --- Use Consistent Warehouse Delay ---
    parser.add_argument("--warehouse-delay", type=int, default=1, help="Warehouse release delay (days)")
    # -------------------------------------
    parser.add_argument("--allocation-batch", type=int, default=1, help="Allocation batch frequency (days, 1=daily)")
    parser.add_argument("--use-blockchain", action="store_true", default=False, help="Enable blockchain integration")
    parser.add_argument("--use-llm", action="store_true", default=False, help="Enable LLM-powered agents (default: use rule-based logic)")
    
    args = parser.parse_args()

    if not args.use_colors: console.no_color = True

    # Create timestamped output folder
    output_folder_path = f"{args.folder}_{timestamp}_regions{args.regions}_drugs{args.drugs}_days{args.days}"
    if args.use_llm: output_folder_path += "_llm"
    else: output_folder_path += "_rules"
    if args.use_blockchain: output_folder_path += "_blockchain"
    
    sim_mode_title = "LLM-Powered" if args.use_llm else "Rule-Based"
    console.print(Panel(f"[bold white]🦠 PANDEMIC SUPPLY CHAIN SIMULATION ({sim_mode_title}) 🦠[/]", border_style="blue", expand=False, padding=(1,2)))
    
    # Config Table
    config_table = Table(title="Simulation Configuration", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan"); config_table.add_column("Value", style="white")
    config_table.add_row("Regions", str(args.regions)); config_table.add_row("Drugs", str(args.drugs))
    config_table.add_row("Simulation Days", str(args.days)); config_table.add_row("Pandemic Severity", f"{args.severity:.2f}")
    config_table.add_row("Disruption Probability Factor", f"{args.disrupt_prob:.2f}")
    config_table.add_row("Warehouse Delay", f"{args.warehouse_delay} days")
    config_table.add_row("Allocation Batch Frequency", f"{args.allocation_batch} days" if args.allocation_batch > 1 else "Daily")
    config_table.add_row("Agent Logic", "[cyan]LLM-Powered[/]" if args.use_llm else "[magenta]Rule-Based[/]")
    if args.use_llm:
        config_table.add_row("  LLM Model", args.model)
    config_table.add_row("Visualizations", "Enabled" if args.visualize else "Disabled")
    config_table.add_row("Verbose Output", "Enabled" if args.verbose else "Disabled")
    config_table.add_row("Output Folder", output_folder_path)
    config_table.add_row("Blockchain", "[bold green]Enabled[/]" if args.use_blockchain else "Disabled")
    if args.use_blockchain:
         config_table.add_row("  Node URL", NODE_URL)
         config_table.add_row("  Contract Address", CONTRACT_ADDRESS or "[red]Not Set[/]")
         config_table.add_row("  ABI Path", CONTRACT_ABI_PATH)
         config_table.add_row("  Signer Key Loaded", "[green]Yes[/]" if BLOCKCHAIN_PRIVATE_KEY else "[red]No[/]")
    console.print(config_table); console.print()

    # Ensure output folder exists
    output_folder = ensure_folder_exists(console, output_folder_path)

    # --- INITIALIZE BLOCKCHAIN INTERFACE ---
    blockchain_interface_instance = None # Initialize as None
    actual_use_blockchain_flag = False # Flag indicating successful BC init

    if args.use_blockchain:
        console.print("\n[bold cyan]Attempting Blockchain Integration...[/]")
        if BlockchainInterface is None:
             console.print("[bold red]❌ Blockchain support not available (missing dependencies like web3?). Halting.[/]")
             exit(1)
        if not check_blockchain_config(): # Uses function from config.py
             console.print("[bold red]❌ Blockchain configuration incomplete in .env or ABI file missing. Halting.[/]")
             console.print("[bold red]   Please ensure NODE_URL, CONTRACT_ADDRESS, BLOCKCHAIN_PRIVATE_KEY are set and ABI exists.[/]")
             exit(1)
        try:
            # Pass the config values to the interface
            blockchain_interface_instance = BlockchainInterface(
                node_url=NODE_URL,
                contract_address=CONTRACT_ADDRESS,
                contract_abi_path=CONTRACT_ABI_PATH,
                private_key=BLOCKCHAIN_PRIVATE_KEY
            )
            actual_use_blockchain_flag = True # Set flag *only* on success
            console.print(f"[bold green]✓ Connected to Ethereum node and loaded contract.[/]")
        except Exception as e:
            console.print(f"[bold red]❌ FATAL ERROR: Could not initialize Blockchain Interface: {e}[/]")
            console.print("[bold red]   Check node connection, contract address, ABI path, and private key format.[/]")
            console.print("[bold red]   Halting simulation execution.[/]")
            try: save_console_html(console, output_folder=output_folder, filename="simulation_error_report.html")
            except Exception as save_e: console.print(f"[red]Could not save error report: {save_e}[/]")
            exit(1) # Exit the program with a non-zero code indicating error
    else:
        actual_use_blockchain_flag = False
        console.print("\n[yellow]Blockchain integration disabled by command-line argument.[/]")
        console.print("[yellow]Running in simulation-only mode.[/]")
    console.print("-" * 30) # Separator

    # --- Run Simulation ---
    results = run_pandemic_simulation(
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
        blockchain_interface=blockchain_interface_instance, # Pass the created instance
        use_blockchain=actual_use_blockchain_flag, # Use the flag indicating successful init
        use_llm=args.use_llm
    )

    # --- Display Results ---
    if results: # Check if simulation ran successfully
        console.print(Panel("[bold white]Simulation Results Summary[/]", border_style="green", expand=False))
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
             def get_service_color(level): return "green" if level >= 95 else "cyan" if level >= 90 else "yellow" if level >= 80 else "red"
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
             # Adjusted rating thresholds considering impact
             total_impact = sum(results["patient_impact"].values())
             if avg_service >= 98 and total_impact < (25 * args.days): rating, rating_color = "Excellent", "green"
             elif avg_service >= 90 and total_impact < (100 * args.days): rating, rating_color = "Good", "cyan"
             elif avg_service >= 80 and total_impact < (500 * args.days): rating, rating_color = "Fair", "yellow"
             else: rating, rating_color = "Poor", "red"
        console.print(f"\n[bold]Overall Supply Chain Performance:[/] [{rating_color}]{rating}[/]")

    else:
        console.print("[bold red]Simulation did not complete successfully. No results to display.[/]")

    # --- PRINT FINAL BLOCKCHAIN STATE ---
       # --- (DISPLAY BLOCKCHAIN PERFORMANCE METRICS ---
    if actual_use_blockchain_flag and blockchain_interface_instance:
        console.print(Panel("[bold white]Blockchain Performance Metrics[/]", border_style=Colors.BLOCKCHAIN, expand=False))
        bc_metrics = blockchain_interface_instance.get_performance_metrics()

        perf_table = Table(title="Blockchain Interaction Summary", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        perf_table.add_column("Metric", style="cyan", min_width=25)
        perf_table.add_column("Value", style="white", min_width=20)

        # Transactions
        perf_table.add_row("[bold]Transactions[/]", "")
        perf_table.add_row("  Attempted Count", str(bc_metrics['tx_sent_count']))
        perf_table.add_row("  Successful Count", str(bc_metrics['tx_success_count']))
        perf_table.add_row("  Failed Count", str(bc_metrics['tx_failure_count']))
        perf_table.add_row("  Success Rate", f"{bc_metrics['tx_success_rate']:.2f}%" if isinstance(bc_metrics['tx_success_rate'], float) else bc_metrics['tx_success_rate'])
        perf_table.add_row("  Avg. Latency (s)", f"{bc_metrics['tx_latency_avg_s']:.4f}")
        perf_table.add_row("  Max Latency (s)", f"{bc_metrics['tx_latency_max_s']:.4f}")
        perf_table.add_row("  P95 Latency (s)", f"{bc_metrics['tx_latency_p95_s']:.4f}")
        perf_table.add_row("  Total Gas Used", str(bc_metrics['total_gas_used']))
        perf_table.add_row("  Avg Gas / Success Tx", f"{bc_metrics['avg_gas_per_successful_tx']:.0f}")
        perf_table.add_row("  Last Tx Error", str(bc_metrics['last_tx_error']) if bc_metrics['last_tx_error'] else "None")

        # Reads
        perf_table.add_row("[bold]Reads (Calls)[/]", "")
        perf_table.add_row("  Attempted Count", str(bc_metrics['read_call_count']))
        perf_table.add_row("  Failed Count", str(bc_metrics['read_error_count']))
        perf_table.add_row("  Success Rate", f"{bc_metrics['read_success_rate']:.2f}%" if isinstance(bc_metrics['read_success_rate'], float) else bc_metrics['read_success_rate'])
        perf_table.add_row("  Avg. Latency (s)", f"{bc_metrics['read_latency_avg_s']:.4f}")
        perf_table.add_row("  Max Latency (s)", f"{bc_metrics['read_latency_max_s']:.4f}")
        perf_table.add_row("  P95 Latency (s)", f"{bc_metrics['read_latency_p95_s']:.4f}")
        perf_table.add_row("  Last Read Error", str(bc_metrics['last_read_error']) if bc_metrics['last_read_error'] else "None")

        console.print(perf_table)

        # --- CALL BLOCKCHAIN VISUALIZATION ---
        visualize_blockchain_performance(
            blockchain_interface=blockchain_interface_instance,
            output_folder=output_folder,
            console=console
        )
        # -----------------------------------------
    elif args.use_blockchain:
        # Case where blockchain was requested but failed to initialize
        console.print(Panel("[yellow]Blockchain Performance Metrics Not Available (Initialization Failed)[/]", border_style="yellow", expand=False))
    
    # Use the actual_use_blockchain_flag and the instance
    if actual_use_blockchain_flag and blockchain_interface_instance:
        console.print("\n[bold cyan]Querying Final Blockchain State...[/]")
        try:
            # Pass expected number of regions/drugs for query loop
            blockchain_interface_instance.print_contract_state(
                num_regions=args.regions, num_drugs=args.drugs
            )
        except Exception as e:
            console.print(f"[red]Error querying final blockchain state: {e}[/]")

    # Save console output to HTML
    sim_mode_suffix = "_llm" if args.use_llm else "_rules"
    html_filename = f"simulation_report{sim_mode_suffix}" + ("_blockchain" if actual_use_blockchain_flag else "") + ".html"
    save_console_html(console, output_folder=output_folder, filename=html_filename)
    console.print(f"\n[green]Visualizations and report saved to folder: '{output_folder}'[/]")

# --- END OF FILE main.py ---