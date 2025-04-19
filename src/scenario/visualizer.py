"""
Visualization functions for pandemic scenarios with SIR model.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.integrate import solve_ivp


def visualize_epidemic_curves(scenario, output_folder="output", console=None):
    """Visualize epidemic curves by region."""
    output_path = os.path.join(output_folder, 'epidemic_curves.png')
    plt.figure(figsize=(12, 8))
    for region_id, curve in scenario.epidemic_curves.items():
        # Check if curve is valid and has data
        if curve is not None and len(curve) > 0:
            region_name = scenario.regions[region_id]["name"]
            plt.plot(curve, label=region_name)
        else:
            if console: console.print(f"[yellow]Warning: Skipping visualization for empty epidemic curve for region {region_id}[/]")

    plt.xlabel('Day'); plt.ylabel('Active Cases')
    plt.title('SIR Model: Active Cases by Region'); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Epidemic curves visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving epidemic curves visualization: {e}[/]")

def visualize_sir_components(scenario, output_folder="output", console=None):
    """Visualize SIR model components (S, I, R) for each region."""
    num_regions = len(scenario.regions)
    
    # Create a figure with subplots for each region
    fig, axes = plt.subplots(num_regions, 1, figsize=(12, 5 * num_regions), sharex=True)
    
    # If there's only one region, wrap the axis in a list for consistent indexing
    if num_regions == 1:
        axes = [axes]
    
    for region_id, region in enumerate(scenario.regions):
        # Get the active cases from the scenario's epidemic curve
        active_cases = scenario.epidemic_curves[region_id]
        
        # Plot active cases
        axes[region_id].plot(active_cases, 'r-', linewidth=2, label='Active Cases')
        
        # Add region information to the title
        population = region["population"]
        region_name = region["name"]
        region_type = region["type"]
        
        axes[region_id].set_title(f'Region {region_name} ({region_type}, Pop: {population:,})')
        axes[region_id].set_ylabel('Cases')
        axes[region_id].grid(True, alpha=0.3)
        axes[region_id].legend()
    
    # Set common x-label for the bottom subplot
    axes[-1].set_xlabel('Day')
    
    # Add an overall title
    plt.suptitle('SIR Model: Active Cases by Region', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_folder, 'sir_components.png')
    try:
        plt.savefig(output_path)
        plt.close()
        if console: console.print(f"[bold green]✓ SIR components visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving SIR components visualization: {e}[/]")

def visualize_sir_simulation(scenario, selected_region_id=0, output_folder="output", console=None):
    """Visualize detailed SIR simulation for a selected region."""
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the selected region
    if selected_region_id >= len(scenario.regions):
        if console: console.print(f"[yellow]Warning: Region ID {selected_region_id} out of range. Using region 0.[/]")
        selected_region_id = 0
    
    region = scenario.regions[selected_region_id]
    region_name = region["name"]
    population = region["population"]
    
    # Re-simulate the SIR model to get S, I, R components
    # This is a demonstration with approximate parameters - not exactly the same as the simulation
    # For a more accurate representation, you would need to store the full SIR results during generation
    
    # Estimate reasonable SIR parameters based on the epidemic curve
    active_cases = scenario.epidemic_curves[selected_region_id]
    peak_day = np.argmax(active_cases)
    peak_cases = active_cases[peak_day]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot active cases
    axes[0].plot(active_cases, 'r-', linewidth=2, label='Active Cases')
    axes[0].set_ylabel('Cases')
    axes[0].set_title(f'Active COVID-19 Cases in {region_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot estimated SIR curves
    # Note: This is an approximation since we don't have the actual S, I, R values
    
    # Estimate R0 and infectious period based on the active cases curve
    # In a real implementation, you would use the actual parameters from the simulation
    r0 = 2.5  # Reasonable default R0 for COVID-19
    infectious_period = 10  # days
    gamma = 1.0 / infectious_period
    beta = r0 * gamma
    
    # Initial conditions
    days = np.arange(len(active_cases))
    initial_infected = active_cases[0] if active_cases[0] > 0 else 10
    initial_susceptible = population - initial_infected
    initial_recovered = 0
    
    # Run a simplified SIR model for visualization
    def sir_model(t, y):
        S, I, R = y
        N = S + I + R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    # Solve the differential equations
    solution = solve_ivp(
        sir_model,
        [0, len(days)-1],
        [initial_susceptible, initial_infected, initial_recovered],
        t_eval=days,
        method='RK45'
    )
    
    S = solution.y[0]
    I = solution.y[1]
    R = solution.y[2]
    
    # Plot percentage of population in each compartment
    axes[1].plot(days, S/population*100, 'b-', label='Susceptible')
    axes[1].plot(days, I/population*100, 'r-', label='Infected')
    axes[1].plot(days, R/population*100, 'g-', label='Recovered')
    axes[1].set_ylabel('Percentage of Population')
    axes[1].set_title(f'SIR Model Components as Percentage of Population (R0≈{r0})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot new cases per day
    # Calculate new cases as the decrease in susceptible population
    new_cases = np.zeros_like(days, dtype=float)
    for i in range(1, len(days)):
        new_cases[i] = max(0, S[i-1] - S[i])
    
    # If day 0 has initial infected, show that as new cases
    new_cases[0] = initial_infected
    
    # Plot new cases
    axes[2].bar(days, new_cases, alpha=0.7, color='orange', label='New Cases')
    axes[2].set_xlabel('Day')
    axes[2].set_ylabel('Number of New Cases')
    axes[2].set_title('Estimated New COVID-19 Cases per Day')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_folder, f'sir_simulation_region_{selected_region_id}.png')
    try:
        plt.savefig(output_path)
        plt.close()
        if console: console.print(f"[bold green]✓ SIR simulation visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving SIR simulation visualization: {e}[/]")


def visualize_drug_demand(scenario, output_folder="output", console = None):
    """Visualize drug demand by region."""
    output_path = os.path.join(output_folder, 'drug_demand.png')
    num_drugs = len(scenario.drugs); num_regions = len(scenario.regions)
    fig, axes = plt.subplots(num_drugs, 1, figsize=(12, 4 * num_drugs), sharex=True, squeeze=False) # Ensure axes is always 2D
    for drug_id, drug in enumerate(scenario.drugs):
        ax = axes[drug_id, 0] # Access subplot correctly
        for region_id in range(num_regions):
            region_name = scenario.regions[region_id]["name"]
            demand_curve = [scenario.get_daily_drug_demand(day, region_id, drug_id) for day in range(scenario.scenario_length)]
            ax.plot(demand_curve, label=region_name)
        ax.set_title(f'Daily Demand for {drug["name"]} (Crit: {drug["criticality"]})')
        ax.set_ylabel('Units Required'); ax.grid(True, alpha=0.3); ax.legend()
    plt.xlabel('Day'); plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Drug demand visualization saved to '{output_path}'[/]")
    except Exception as e:
        if console: console.print(f"[bold red]Error saving drug demand visualization: {e}[/]")

def visualize_disruptions(scenario, output_folder="output", console=None):
    """Visualize manufacturing and transportation disruptions."""
    output_path = os.path.join(output_folder, 'disruptions.png')
    manufacturing_disruptions = [d for d in scenario.disruptions if d["type"] == "manufacturing"]
    transportation_disruptions = [d for d in scenario.disruptions if d["type"] == "transportation"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    if manufacturing_disruptions:
        for i, disruption in enumerate(manufacturing_disruptions):
            drug_name = scenario.drugs[disruption["drug_id"]]["name"]
            start = disruption["start_day"]
            end = disruption["end_day"]
            severity = disruption["severity"]
            ax1.barh(i, end - start + 1, left=start, height=0.8, color=plt.cm.Reds(severity), alpha=0.7) # Added +1 to duration for visualization
            ax1.text(start + (end - start + 1) / 2, i, f"{severity:.2f}", ha='center', va='center', color='black')
        ax1.set_yticks(range(len(manufacturing_disruptions)))
        ax1.set_yticklabels([scenario.drugs[d["drug_id"]]["name"] for d in manufacturing_disruptions])
    ax1.set_title('Manufacturing Disruptions'); ax1.set_xlabel('Day')
    if transportation_disruptions:
        for i, disruption in enumerate(transportation_disruptions):
            region_name = scenario.regions[disruption["region_id"]]["name"]
            start = disruption["start_day"]
            end = disruption["end_day"]
            severity = disruption["severity"]
            ax2.barh(i, end - start + 1, left=start, height=0.8, color=plt.cm.Blues(severity), alpha=0.7) # Added +1 to duration for visualization
            ax2.text(start + (end - start + 1) / 2, i, f"{severity:.2f}", ha='center', va='center', color='black')
        ax2.set_yticks(range(len(transportation_disruptions)))
        ax2.set_yticklabels([scenario.regions[d["region_id"]]["name"] for d in transportation_disruptions])
    ax2.set_title('Transportation Disruptions'); ax2.set_xlabel('Day')
    plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        if console: console.print(f"[bold green]✓ Disruptions visualization saved to '{output_path}'[/]")
    except Exception as e:
         if console: console.print(f"[bold red]Error saving disruptions visualization: {e}[/]")