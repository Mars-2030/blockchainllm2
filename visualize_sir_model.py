"""
Visualization script for SIR model.

This script provides detailed visualizations of the SIR epidemic model
to help understand the underlying dynamics of the pandemic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from matplotlib.gridspec import GridSpec

def sir_model(t, y, beta, gamma):
    """SIR model differential equations."""
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def visualize_r0_comparison():
    """Visualize how different R0 values affect the epidemic curve."""
    # Population size
    N = 1000000
    
    # Initial conditions
    I0 = 100  # Start with 100 infected
    S0 = N - I0
    R0 = 0
    
    # Time span for the simulation
    t_max = 180  # days
    t = np.linspace(0, t_max, t_max + 1)
    
    # Different R0 values to compare
    r0_values = [1.1, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    # Infectious period (days)
    infectious_period = 10  # gamma = 1/infectious_period
    gamma = 1.0 / infectious_period
    
    plt.figure(figsize=(15, 10))
    
    # Create a GridSpec for more complex layout
    gs = GridSpec(2, 2, figure=plt.gcf(), height_ratios=[1, 1])
    
    # Infected curves plot
    ax1 = plt.subplot(gs[0, :])
    
    # Peak and timing plot
    ax2 = plt.subplot(gs[1, 0])
    
    # Final size plot
    ax3 = plt.subplot(gs[1, 1])
    
    # Arrays to store peak values and timing
    peak_values = []
    peak_times = []
    final_sizes = []
    
    # Simulate for each R0 value
    for r0 in r0_values:
        beta = r0 * gamma
        
        # Solve the ODE system
        solution = solve_ivp(
            lambda t, y: sir_model(t, y, beta, gamma),
            [0, t_max],
            [S0, I0, R0],
            t_eval=t,
            method='RK45'
        )
        
        S = solution.y[0]
        I = solution.y[1]
        R = solution.y[2]
        
        # Plot infected curve
        ax1.plot(solution.t, I, label=f'R0 = {r0}')
        
        # Store peak value and time
        peak_idx = np.argmax(I)
        peak_values.append(I[peak_idx])
        peak_times.append(solution.t[peak_idx])
        
        # Final size (total infected)
        final_sizes.append(R[-1])
    
    # Plot peak values vs R0
    ax2.bar(r0_values, peak_values, width=0.3, alpha=0.6, color='red')
    ax2.set_xlabel('R0')
    ax2.set_ylabel('Peak Infected')
    ax2.set_title('Peak Epidemic Size vs R0')
    ax2.set_xticks(r0_values)
    ax2.grid(alpha=0.3)
    
    # Plot final sizes vs R0
    ax3.bar(r0_values, final_sizes, width=0.3, alpha=0.6, color='blue')
    ax3.set_xlabel('R0')
    ax3.set_ylabel('Total Infected (Final)')
    ax3.set_title('Final Epidemic Size vs R0')
    ax3.set_xticks(r0_values)
    ax3.grid(alpha=0.3)
    
    # Configure the main plot
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Number of Infected')
    ax1.set_title('Infected Curves for Different R0 Values')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/r0_comparison.png')
    plt.close()
    print(f"R0 comparison visualization saved to 'output/r0_comparison.png'")

def visualize_multi_wave_dynamics():
    """Visualize multi-wave dynamics with SIR model."""
    # Population size
    N = 1000000
    
    # Time span
    t_max = 180
    
    # Set up the figure
    plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 1, figure=plt.gcf(), height_ratios=[1, 1, 1])
    
    # Wave 1 plot
    ax1 = plt.subplot(gs[0])
    ax1.set_title('Wave 1: Initial Outbreak')
    
    # Wave 2 plot
    ax2 = plt.subplot(gs[1])
    ax2.set_title('Wave 2: Second Outbreak from Remaining Susceptibles')
    
    # Combined waves plot
    ax3 = plt.subplot(gs[2])
    ax3.set_title('Combined Active Cases with Multiple Waves')
    
    # Wave parameters
    wave_params = [
        {  # Wave 1
            'start_day': 0,
            'beta': 0.3,  # High transmission
            'gamma': 0.1,  # 10-day infectious period
            'initial_susceptible': N - 100,
            'initial_infected': 100,
            'initial_recovered': 0
        },
        {  # Wave 2
            'start_day': 90,  # Starts after day 90
            'beta': 0.35,  # Even higher transmission (e.g., new variant)
            'gamma': 0.12,  # Slightly shorter infectious period
            'initial_infected': 10,  # Small number of new infections
            # initial_susceptible and initial_recovered will be calculated based on Wave 1 results
        }
    ]
    
    # Simulate Wave 1
    wave1 = wave_params[0]
    solution1 = solve_ivp(
        lambda t, y: sir_model(t, y, wave1['beta'], wave1['gamma']),
        [0, t_max],
        [wave1['initial_susceptible'], wave1['initial_infected'], wave1['initial_recovered']],
        t_eval=np.arange(t_max + 1),
        method='RK45'
    )
    
    S1, I1, R1 = solution1.y
    
    # Plot Wave 1 components
    ax1.plot(solution1.t, S1, 'b-', label='Susceptible')
    ax1.plot(solution1.t, I1, 'r-', label='Infected')
    ax1.plot(solution1.t, R1, 'g-', label='Recovered')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Initialize arrays for combined simulation
    active_cases = np.zeros(t_max + 1)
    active_cases[:len(I1)] = I1  # Add Wave 1 active cases
    
    # Get state at the start of Wave 2
    wave2 = wave_params[1]
    start_day = wave2['start_day']
    
    # Calculate remaining susceptible population
    remaining_susceptible = S1[start_day] * 0.6  # Assume some recovered lost immunity
    previously_recovered = R1[start_day]
    newly_susceptible = previously_recovered * 0.3  # Some recovered become susceptible again
    
    # Set up initial conditions for Wave 2
    wave2['initial_susceptible'] = remaining_susceptible + newly_susceptible
    wave2['initial_recovered'] = previously_recovered - newly_susceptible
    
    # Simulate Wave 2
    solution2 = solve_ivp(
        lambda t, y: sir_model(t, y, wave2['beta'], wave2['gamma']),
        [0, t_max - start_day],
        [wave2['initial_susceptible'], wave2['initial_infected'], wave2['initial_recovered']],
        t_eval=np.arange(t_max - start_day + 1),
        method='RK45'
    )
    
    S2, I2, R2 = solution2.y
    
    # Plot Wave 2 components
    t2 = np.arange(start_day, start_day + len(S2))
    ax2.plot(t2, S2, 'b-', label='Susceptible')
    ax2.plot(t2, I2, 'r-', label='Infected')
    ax2.plot(t2, R2, 'g-', label='Recovered')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Add Wave 2 to combined active cases
    for i, day in enumerate(range(start_day, start_day + len(I2))):
        if day < len(active_cases):
            active_cases[day] += I2[i]
    
    # Plot combined active cases
    ax3.plot(np.arange(t_max + 1), active_cases, 'r-', linewidth=2, label='Active Cases')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # Set common labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Days')
        ax.set_ylabel('Population')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/multi_wave_dynamics.png')
    plt.close()
    print(f"Multi-wave dynamics visualization saved to 'output/multi_wave_dynamics.png'")

def visualize_region_comparison():
    """Visualize SIR model in different region types."""
    
    # Define region types with different parameters
    regions = [
        {
            "name": "Urban",
            "population": 1000000,
            "R0": 3.0,        # Higher transmission in dense areas
            "initial_infected_pct": 0.001  # 0.1% initially infected
        },
        {
            "name": "Suburban",
            "population": 500000,
            "R0": 2.5,        # Moderate transmission
            "initial_infected_pct": 0.0005  # 0.05% initially infected
        },
        {
            "name": "Rural",
            "population": 100000,
            "R0": 2.0,        # Lower transmission in less dense areas
            "initial_infected_pct": 0.0002  # 0.02% initially infected
        },
        {
            "name": "Remote",
            "population": 20000,
            "R0": 1.5,        # Lowest transmission due to isolation
            "initial_infected_pct": 0.0001  # 0.01% initially infected
        }
    ]
    
    # Time span for the simulation
    t_max = 180
    t = np.linspace(0, t_max, t_max + 1)
    
    # Infectious period (days)
    infectious_period = 10
    gamma = 1.0 / infectious_period
    
    plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Prepare axes
    ax1 = plt.subplot(gs[0, 0])  # Active cases
    ax2 = plt.subplot(gs[0, 1])  # Percent infected
    ax3 = plt.subplot(gs[1, :])  # All regions active cases
    
    # Colors for regions
    colors = ['red', 'blue', 'green', 'purple']
    
    # Simulate each region
    for i, region in enumerate(regions):
        # Calculate parameters
        N = region["population"]
        R0 = region["R0"]
        beta = R0 * gamma
        
        # Initial conditions
        I0 = int(N * region["initial_infected_pct"])
        S0 = N - I0
        R0_count = 0
        
        # Solve the ODE system
        solution = solve_ivp(
            lambda t, y: sir_model(t, y, beta, gamma),
            [0, t_max],
            [S0, I0, R0_count],
            t_eval=t,
            method='RK45'
        )
        
        S = solution.y[0]
        I = solution.y[1]
        R = solution.y[2]
        
        # Plot in the first appropriate subplot
        if i == 0:  # Urban
            ax = ax1
        else:
            ax = ax2
        
        # Plot SIR components for this region
        ax.plot(solution.t, S/N*100, 'b-', label='Susceptible')
        ax.plot(solution.t, I/N*100, 'r-', label='Infected')
        ax.plot(solution.t, R/N*100, 'g-', label='Recovered')
        ax.set_title(f'{region["name"]} Region (Pop: {N:,})')
        ax.set_xlabel('Days')
        ax.set_ylabel('Percent of Population')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot active cases on the combined plot
        ax3.plot(solution.t, I, color=colors[i], label=f'{region["name"]} (R0: {R0})')
    
    # Configure the combined plot
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Active Cases')
    ax3.set_title('Active Cases Comparison Across Region Types')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/region_comparison.png')
    plt.close()
    print(f"Region comparison visualization saved to 'output/region_comparison.png'")

if __name__ == "__main__":
    print("Generating SIR model visualizations...")
    
    # Generate all visualizations
    visualize_r0_comparison()
    visualize_multi_wave_dynamics()
    visualize_region_comparison()
    
    print("All SIR model visualizations complete!")