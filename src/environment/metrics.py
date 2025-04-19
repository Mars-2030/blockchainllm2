"""
Metrics tracking and visualization for the pandemic supply chain.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def track_service_levels(environment):
    """Track percentage of demand met over time."""
    service_levels = []
    # Group history by day first
    demands_by_day = {}
    for d in environment.demand_history:
        day = d["day"]
        if day not in demands_by_day: demands_by_day[day] = []
        demands_by_day[day].append(d)

    stockouts_by_day = {}
    for s in environment.stockout_history:
        day = s["day"]
        if day not in stockouts_by_day: stockouts_by_day[day] = []
        stockouts_by_day[day].append(s)

    # Calculate daily service level
    days = sorted(list(set(demands_by_day.keys()) | set(stockouts_by_day.keys())))
    for day in days:
         day_demands = demands_by_day.get(day, [])
         day_stockouts = stockouts_by_day.get(day, [])

         total_demand_for_day = sum(d["demand"] for d in day_demands)
         total_unfulfilled_for_day = sum(s["unfulfilled"] for s in day_stockouts)

         if total_demand_for_day > 0:
             # Ensure service level is capped between 0 and 100
             level = 100 * (1 - total_unfulfilled_for_day / total_demand_for_day)
             service_level = max(0, min(100, level)) # Cap between 0 and 100
         else:
             service_level = 100 # 100% if no demand

         service_levels.append({"day": day, "service_level": service_level})

    return service_levels

def visualize_service_levels(environment, output_folder="output", console=None):
    """Visualize service levels over time."""
    service_levels = track_service_levels(environment)
    if not service_levels:
        console.print("[yellow]No service level data available to visualize.[/]")
        return
    output_path = os.path.join(output_folder, 'service_levels.png')
    days = [item["day"] for item in service_levels]
    service_level_values = [item["service_level"] for item in service_levels]
    plt.figure(figsize=(12, 6))
    plt.plot(days, service_level_values, marker='.', linestyle='-', color='blue', markersize=4)
    plt.axhline(y=95, color='green', linestyle='--', label='Excellent (95%)')
    plt.axhline(y=90, color='orange', linestyle='--', label='Good (90%)')
    plt.axhline(y=80, color='red', linestyle='--', label='Critical (80%)')
    plt.xlabel('Day'); plt.ylabel('Service Level (%)')
    plt.title('Daily Service Level Over Time (% of Demand Met)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.ylim(0, 105)
    plt.tight_layout()
    try:
        plt.savefig(output_path); plt.close()
        console.print(f"[bold green]✓ Service levels visualization saved to '{output_path}'[/]")
    except Exception as e:
         console.print(f"[bold red]Error saving service levels visualization: {e}[/]")

def visualize_performance(environment, output_folder="output", console=None):
    """Visualize supply chain performance metrics."""
    output_path = os.path.join(output_folder, 'supply_chain_performance.png')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Stockouts heatmap
    stockout_data = [{"Drug": environment.scenario.drugs[d]["name"], 
                      "Region": environment.scenario.regions[r]["name"], 
                      "Stockouts": environment.stockouts[d][r]}
                     for d in range(environment.num_drugs) 
                     for r in range(environment.num_regions)]
    if stockout_data:
        stockout_df = pd.DataFrame(stockout_data)
        try:
             stockout_pivot = stockout_df.pivot(index="Drug", columns="Region", values="Stockouts")
             sns.heatmap(stockout_pivot, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0], linewidths=.5)
             axes[0].set_title("Stockout Days by Drug and Region")
        except ValueError as e:
             axes[0].text(0.5, 0.5, 'No stockout data to plot or pivot failed.', horizontalalignment='center', verticalalignment='center')
             axes[0].set_title("Stockout Days by Drug and Region")
             if console: console.print(f"[yellow]Could not generate stockout heatmap: {e}[/]")

    else:
        axes[0].text(0.5, 0.5, 'No stockout data available.', horizontalalignment='center', verticalalignment='center')
        axes[0].set_title("Stockout Days by Drug and Region")


    # 2. Unfulfilled demand bar chart
    unfulfilled_data = []
    for drug_id in range(environment.num_drugs):
        for region_id in range(environment.num_regions):
             unfulfilled = environment.unfulfilled_demand[drug_id][region_id]
             total = environment.total_demand[drug_id][region_id]
             percent_unfulfilled = (unfulfilled / total) * 100 if total > 0 else 0
             unfulfilled_data.append({
                 "Drug": environment.scenario.drugs[drug_id]["name"],
                 "Region": environment.scenario.regions[region_id]["name"],
                 "Percent Unfulfilled": percent_unfulfilled
             })

    if unfulfilled_data:
         unfulfilled_df = pd.DataFrame(unfulfilled_data)
         # Using seaborn directly for potentially better grouping/handling
         sns.barplot(data=unfulfilled_df, x="Drug", y="Percent Unfulfilled", hue="Region", ax=axes[1], errorbar=None)
         axes[1].set_title("Percentage of Unfulfilled Demand by Drug and Region")
         axes[1].set_ylabel("Percent Unfulfilled (%)")
         axes[1].tick_params(axis='x', rotation=45) # Rotate drug names if needed
         axes[1].legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
         axes[1].grid(True, axis="y", alpha=0.3)
    else:
         axes[1].text(0.5, 0.5, 'No unfulfilled demand data available.', horizontalalignment='center', verticalalignment='center')
         axes[1].set_title("Percentage of Unfulfilled Demand by Drug and Region")


    # 3. Patient impact bar chart
    impact_data = [{"Region": environment.scenario.regions[r]["name"], "Patient Impact": environment.patient_impact[r]}
                   for r in range(environment.num_regions)]
    if impact_data:
         impact_df = pd.DataFrame(impact_data)
         # Assign hue explicitly for newer seaborn versions
         sns.barplot(x="Region", y="Patient Impact", data=impact_df, ax=axes[2], palette="viridis", hue="Region", legend=False)
         axes[2].set_title("Patient Impact Score by Region")
         axes[2].set_ylabel("Impact Score (higher is worse)")
         axes[2].grid(True, axis="y", alpha=0.3)
    else:
         axes[2].text(0.5, 0.5, 'No patient impact data available.', horizontalalignment='center', verticalalignment='center')
         axes[2].set_title("Patient Impact Score by Region")


    plt.tight_layout()
    try:
         plt.savefig(output_path); plt.close()
         if console: console.print(f"[bold green]✓ Supply chain performance visualization saved to '{output_path}'[/]")
    except Exception as e:
          if console: console.print(f"[bold red]Error saving performance visualization: {e}[/]")

def visualize_inventory_levels(environment, output_folder="output", console = None):
    """Visualize inventory levels including warehouse."""
    output_path = os.path.join(output_folder, 'inventory_levels.png')
    if not environment.inventory_history or not environment.warehouse_history:
        console.print("[yellow]No inventory/warehouse history data available for visualization.[/]")
        return

    days = sorted(environment.inventory_history.keys())
    if not days: return # No history recorded

    fig, axes = plt.subplots(environment.num_drugs, 1, figsize=(14, 5 * environment.num_drugs), sharex=True, squeeze=False)

    colors = {'warehouse': 'cyan', 'manufacturer': 'blue', 'distributor': 'green', 'hospital': 'magenta'}

    for drug_id in range(environment.num_drugs):
        ax = axes[drug_id, 0]
        drug_info = environment.scenario.drugs[drug_id]
        drug_name = drug_info.get("name", f"Drug-{drug_id}")
        criticality = drug_info.get("criticality", "Unknown")

        # Plot Warehouse
        warehouse_inv = [environment.warehouse_history.get(day, {}).get(drug_id, 0) for day in days]
        ax.plot(days, warehouse_inv, label="Warehouse", color=colors['warehouse'], linestyle="-", lw=2)

        # Plot Manufacturer
        manu_inv = [environment.inventory_history.get(day, {}).get(drug_id, {}).get(0, 0) for day in days]
        ax.plot(days, manu_inv, label="Manufacturer", color=colors['manufacturer'], linestyle="-", lw=2)

        # Plot Distributors (Combined)
        dist_inv_total = [0] * len(days)
        for r_id in range(environment.num_regions):
            dist_node_id = r_id + 1
            dist_inv = [environment.inventory_history.get(day, {}).get(drug_id, {}).get(dist_node_id, 0) for day in days]
            # ax.plot(days, dist_inv, label=f"Dist {r_id}", color=colors['distributor'], linestyle="--", alpha=0.5) # Optional: plot individual distributors
            dist_inv_total = [x + y for x, y in zip(dist_inv_total, dist_inv)]
        ax.plot(days, dist_inv_total, label="All Distributors", color=colors['distributor'], linestyle="-", lw=2)

        # Plot Hospitals (Combined)
        hosp_inv_total = [0] * len(days)
        for r_id in range(environment.num_regions):
            hosp_node_id = environment.num_regions + 1 + r_id
            hosp_inv = [environment.inventory_history.get(day, {}).get(drug_id, {}).get(hosp_node_id, 0) for day in days]
            hosp_inv_total = [x + y for x, y in zip(hosp_inv_total, hosp_inv)]
        ax.plot(days, hosp_inv_total, label="All Hospitals", color=colors['hospital'], linestyle="-", lw=2)

        # Plot Total System Inventory
        total_system_inv = [w + m + d + h for w, m, d, h in zip(warehouse_inv, manu_inv, dist_inv_total, hosp_inv_total)]
        ax.plot(days, total_system_inv, label="Total System", color="black", linestyle=":", lw=2)


        # Batch Allocation Markers
        batch_freq = getattr(environment, 'allocation_batch_frequency', 1)
        if batch_freq > 1:
            batch_days = [d for d in days if d > 0 and d % batch_freq == 0] # Exclude day 0
            if batch_days:
                 # Get ylim *after* plotting data
                 y_min, y_max = ax.get_ylim()
                 # Extend y_max slightly for visibility if lines go near top
                 if y_max < 1: y_max = 1 # Avoid too small range
                 ax.vlines(batch_days, ymin=y_min, ymax=y_max * 1.05, color='grey', linestyle='--', alpha=0.6, label=f'Batch Alloc (freq={batch_freq}d)')
                 # Set ylim again after adding vlines
                 ax.set_ylim(bottom=y_min, top = y_max * 1.05)


        ax.set_title(f"Inventory: {drug_name} (Crit: {criticality})")
        ax.set_ylabel("Units"); ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
        # Ensure y starts at 0 unless data is huge
        if max(total_system_inv or [0]) < 1e6: # Avoid huge scales forcing ymin
             current_ylim = ax.get_ylim()
             ax.set_ylim(bottom=0, top=current_ylim[1]) # Set bottom to 0


    plt.suptitle("Supply Chain Inventory Levels Over Time", fontsize=16, y=1.02) # Adjust title pos slightly higher
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

    delay = getattr(environment, 'warehouse_release_delay', 0)
    freq = getattr(environment, 'allocation_batch_frequency', 1)
    fig.text(0.5, 0.01, f"Warehouse Delay: {delay}d, Allocation Batch Freq: {freq}d", ha='center', fontsize=10, style='italic')

    try:
         plt.savefig(output_path); plt.close()
         console.print(f"[bold green]✓ Inventory levels visualization saved to '{output_path}'[/]")
    except Exception as e:
         console.print(f"[bold red]Error saving inventory levels visualization: {e}[/]")

    # Call warehouse flow visualization
    visualize_warehouse_flow(environment, output_folder, console = console)

def visualize_warehouse_flow(environment, output_folder="output", console = None):
    """Visualize warehouse -> manufacturer flow."""
    output_path = os.path.join(output_folder, 'warehouse_flow.png')
    if not environment.warehouse_release_history:
        return

    releases_by_day = {}
    max_day = 0
    for release in environment.warehouse_release_history:
        day = release["day"]
        max_day = max(max_day, day)
        if day not in releases_by_day: releases_by_day[day] = []
        releases_by_day[day].append(release)

    # Include days up to the end of the simulation for the plot range
    plot_days = max(max_day, environment.current_day) if environment.current_day > 0 else max_day
    if plot_days == 0 and not releases_by_day: return # No releases happened

    days_list = list(range(plot_days + 1))
    cumulative_releases = {drug_id: np.zeros(len(days_list)) for drug_id in range(environment.num_drugs)}

    for day_idx, day in enumerate(days_list):
        if day_idx > 0: # Start accumulating from previous day
            for drug_id in range(environment.num_drugs):
                cumulative_releases[drug_id][day_idx] = cumulative_releases[drug_id][day_idx - 1]

        if day in releases_by_day:
             for release in releases_by_day[day]:
                  drug_id = release["drug_id"]
                  # Add release amount only to the current day's cumulative value
                  cumulative_releases[drug_id][day_idx] += release["amount"]


    fig, ax = plt.subplots(figsize=(12, 6)) # Reduced height slightly
    drug_colors = plt.cm.tab10(np.linspace(0, 1, environment.num_drugs)) # Use consistent colors

    for drug_id in range(environment.num_drugs):
        drug_info = environment.scenario.drugs[drug_id]
        drug_name = drug_info.get("name", f"Drug-{drug_id}")
        crit = drug_info.get("criticality", "?")
        ax.plot(days_list, cumulative_releases[drug_id], label=f"{drug_name} (Crit: {crit})", lw=2, color=drug_colors[drug_id])


    ax.set_title("Cumulative Inventory Released from Warehouse to Manufacturer")
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative Units Released")
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left')

    current_ylim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_ylim[1]) # Ensure y starts at 0


    delay = getattr(environment, 'warehouse_release_delay', 0)
    if delay > 0:
        ax.axvspan(0, delay, color='lightgray', alpha=0.3, label=f'Initial {delay}-day delay')
        # Need to call legend again if adding labels after plotting
        handles, labels = ax.get_legend_handles_labels()
        # Avoid duplicate labels
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels[l] = h
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')


    plt.figtext(0.5, 0.01, f"Warehouse Release Delay: {delay} days", ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout

    try:
         plt.savefig(output_path); plt.close()
         console.print(f"[bold green]✓ Warehouse flow visualization saved to '{output_path}'[/]")
    except Exception as e:
         console.print(f"[bold red]Error saving warehouse flow visualization: {e}[/]")