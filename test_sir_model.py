"""
Test script for SIR model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def sir_model(t, y, beta, gamma):
    """SIR model differential equations."""
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def visualize_basic_sir():
    """Visualize a basic SIR model."""
    # Total population
    N = 1000000
    
    # Initial conditions: 1 infected, rest susceptible
    I0 = 1
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    
    # Parameters
    beta = 0.3  # infection rate
    gamma = 0.1  # recovery rate (1/infectious period)
    R0_value = beta/gamma
    print(f"Basic reproduction number (R0): {R0_value}")
    
    # Time vector
    t_max = 180
    t = np.linspace(0, t_max, t_max + 1)
    
    # Solve the ODE system
    solution = solve_ivp(
        lambda t, y: sir_model(t, y, beta, gamma),
        [0, t_max],
        y0,
        t_eval=t,
        method='RK45'
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(solution.t, solution.y[0], 'b-', label='Susceptible')
    plt.plot(solution.t, solution.y[1], 'r-', label='Infected')
    plt.plot(solution.t, solution.y[2], 'g-', label='Recovered')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title(f'SIR Model (beta={beta}, gamma={gamma}, R0={R0_value:.2f})')
    plt.legend()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/basic_sir_model.png')
    plt.close()
    print(f"Basic SIR model visualization saved to 'output/basic_sir_model.png'")

def simulate_sir_with_waves(population, num_days, num_waves=2):
    """Simulate SIR model with multiple infection waves."""
    
    # Initialize arrays for tracking
    total_cases = np.zeros(num_days)
    active_cases = np.zeros(num_days)
    new_cases_per_day = np.zeros(num_days)
    
    plt.figure(figsize=(15, 10))
    
    for wave in range(num_waves):
        # Randomly determine wave parameters
        wave_start = np.random.randint(0, num_days - 100)
        
        # For subsequent waves, only a portion of previously unexposed are susceptible
        if wave > 0:
            remaining_population = population - total_cases[wave_start]
            initial_susceptible = remaining_population * np.random.uniform(0.7, 0.95)
        else:
            initial_susceptible = population * 0.99  # Almost everyone starts susceptible
        
        # Set SIR parameters for this wave
        # Different waves can have different R0 values
        r0 = np.random.uniform(1.5, 3.5)
        
        # Recovery rate (1/gamma is the infectious period in days)
        infectious_period = np.random.uniform(5, 14)
        gamma = 1.0 / infectious_period
        
        # Calculate beta from R0 and gamma
        beta = r0 * gamma
        
        # Initial conditions [S, I, R]
        # Start with a small number of infections
        wave_severity = 0.1 if wave == 0 else 0.05
        initial_infected = max(1, population * 0.0001 * wave_severity)
        initial_recovered = 0
        
        # Ensure initial values don't exceed population
        initial_susceptible = min(initial_susceptible, population - initial_infected - initial_recovered)
        
        y0 = [initial_susceptible, initial_infected, initial_recovered]
        
        # Time points for the wave (from wave start to end of simulation)
        t_span = [0, num_days - wave_start]
        t_eval = np.arange(0, num_days - wave_start)
        
        # Solve the SIR differential equations
        solution = solve_ivp(
            lambda t, y: sir_model(t, y, beta, gamma),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45'
        )
        
        # Extract the solution
        S = solution.y[0]
        I = solution.y[1]
        R = solution.y[2]
        
        # Plot the individual wave components
        wave_label = f'Wave {wave+1}'
        t_plot = [wave_start + i for i in range(len(t_eval))]
        plt.plot(t_plot, I, label=f'{wave_label} - Active Cases', linestyle='--')
        
        # Add this wave's contribution to total cases
        for i in range(len(t_eval)):
            day_idx = wave_start + i
            if day_idx < num_days:
                # Active cases are directly from the I compartment
                active_cases[day_idx] += I[i]
                
                # Calculate new cases for this day by looking at the decrease in susceptible
                if i > 0:
                    new_infections = max(0, S[i-1] - S[i])
                    new_cases_per_day[day_idx] += new_infections
                elif i == 0 and initial_infected > 0:
                    # For the first day, use initial infected as new cases
                    new_cases_per_day[day_idx] += initial_infected
                    
                # Accumulate total cases
                if day_idx > 0:
                    total_cases[day_idx] = total_cases[day_idx-1] + new_cases_per_day[day_idx]
                else:
                    total_cases[day_idx] = new_cases_per_day[day_idx]
    
    # Plot the aggregated results
    plt.plot(range(num_days), active_cases, 'r-', linewidth=3, label='Total Active Cases')
    plt.plot(range(num_days), total_cases, 'b-', linewidth=3, label='Cumulative Cases')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Cases')
    plt.title(f'SIR Model with {num_waves} Waves - Population {population:,}')
    plt.legend()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/multi_wave_sir_model.png')
    plt.close()
    print(f"Multi-wave SIR model visualization saved to 'output/multi_wave_sir_model.png'")
    
    return {
        "active_cases": active_cases,
        "total_cases": total_cases,
        "new_cases_per_day": new_cases_per_day
    }

def test_multi_region_sir():
    """Test multi-region SIR simulations."""
    
    # Define several regions with different characteristics
    regions = [
        {"name": "Urban", "population": 1000000, "type": "Urban"},
        {"name": "Suburban", "population": 500000, "type": "Suburban"},
        {"name": "Rural", "population": 100000, "type": "Rural"},
        {"name": "Remote", "population": 20000, "type": "Remote"}
    ]
    
    num_days = 180
    
    plt.figure(figsize=(15, 10))
    
    # Simulate each region
    for region in regions:
        # Number of waves varies by region type
        if region["type"] == "Urban":
            num_waves = 3
        elif region["type"] == "Suburban":
            num_waves = 2
        else:
            num_waves = 1
            
        # Simulate SIR model with waves
        results = simulate_sir_with_waves(region["population"], num_days, num_waves)
        
        # Plot active cases for this region
        plt.plot(range(num_days), results["active_cases"], label=f"{region['name']} - Active Cases")
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Active Cases')
    plt.title('SIR Model: Active Cases by Region')
    plt.legend()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/multi_region_sir_model.png')
    plt.close()
    print(f"Multi-region SIR model visualization saved to 'output/multi_region_sir_model.png'")

if __name__ == "__main__":
    # Run the tests
    print("Testing basic SIR model...")
    visualize_basic_sir()
    
    print("\nTesting multi-wave SIR model...")
    simulate_sir_with_waves(population=1000000, num_days=180, num_waves=2)
    
    print("\nTesting multi-region SIR model...")
    test_multi_region_sir()
    
    print("\nAll visualizations generated successfully!")