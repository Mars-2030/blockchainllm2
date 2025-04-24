"""
Pandemic scenario generation module with SIR model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os
from scipy.integrate import solve_ivp


# Import from the config module
# from config import console

class PandemicScenarioGenerator:
    """Generates synthetic pandemic scenarios with regions, drugs, epidemic curves, and disruptions using SIR model."""
    
    def __init__(
        self,
        console: None,
        num_regions: int = 5,
        num_drugs: int = 3,
        scenario_length: int = 180,
        pandemic_severity: float = 0.8,
        disruption_probability: float = 0.2
    ):
        self.console = console
        self.num_regions = num_regions
        self.num_drugs = num_drugs
        self.scenario_length = scenario_length
        self.pandemic_severity = pandemic_severity
        self.disruption_probability = disruption_probability
        self.regions = self._generate_regions()
        self.drugs = self._generate_drugs()
        self.epidemic_curves = self._generate_epidemic_curves()
        self.disruptions = self._generate_disruptions()

    def _generate_regions(self) -> List[Dict]:
        regions = []
        region_types = ["Urban", "Suburban", "Rural", "Remote"]
        for i in range(self.num_regions):
            region_type = region_types[i % len(region_types)]
            if region_type == "Urban": population = np.random.randint(500000, 5000000)
            elif region_type == "Suburban": population = np.random.randint(100000, 500000)
            elif region_type == "Rural": population = np.random.randint(20000, 100000)
            else: population = np.random.randint(5000, 20000) # Remote
            if region_type == "Urban": healthcare_capacity = np.random.uniform(3.0, 5.0)
            elif region_type == "Suburban": healthcare_capacity = np.random.uniform(2.0, 4.0)
            elif region_type == "Rural": healthcare_capacity = np.random.uniform(1.5, 3.0)
            else: healthcare_capacity = np.random.uniform(0.5, 2.0) # Remote
            total_capacity = int(population * healthcare_capacity / 1000)
            regions.append({
                "id": i, "name": f"Region-{i+1}", "type": region_type,
                "population": population, "healthcare_capacity": healthcare_capacity,
                "total_beds": total_capacity,
                "transportation_reliability": np.random.uniform(0.7, 0.99)
            })
        return regions

    def _generate_drugs(self) -> List[Dict]:
        drugs = []
        criticality_levels = ["Critical", "High", "Medium", "Low"]
        for i in range(self.num_drugs):
            criticality = criticality_levels[i % len(criticality_levels)]
            if criticality == "Critical": shelf_life = np.random.randint(30, 90)
            elif criticality == "High": shelf_life = np.random.randint(90, 180)
            elif criticality == "Medium": shelf_life = np.random.randint(180, 365)
            else: shelf_life = np.random.randint(365, 730) # Low
            production_complexity = np.random.uniform(0.1, 0.9)
            if criticality == "Critical": substitutability = np.random.uniform(0, 0.3)
            elif criticality == "High": substitutability = np.random.uniform(0.2, 0.5)
            else: substitutability = np.random.uniform(0.4, 0.9)
            if criticality == "Critical": base_demand = np.random.uniform(50, 100)
            elif criticality == "High": base_demand = np.random.uniform(20, 50)
            elif criticality == "Medium": base_demand = np.random.uniform(10, 20)
            else: base_demand = np.random.uniform(1, 10) # Low
            # Production capacities from code15
            # --- ADJUSTED Production Capacities (Interpreted as DAILY) ---
            # Note: Assuming 'base_production' is used as daily capacity limit
            # These values should be high enough to handle surges but not infinite.
            if criticality == "Critical":
                 # Daily capacity significantly higher than peak regional daily demand
                 base_production = np.random.uniform(15000, 30000) # WAS: 150k-300k
            elif criticality == "High":
                 base_production = np.random.uniform(5000, 15000)  # WAS: 50k-150k
            else: # Medium/Low
                 base_production = np.random.uniform(2500, 5000)   # WAS: 25k-50k
            # -------------------------------------------------------------
            drugs.append({
                "id": i, "name": f"Drug-{i+1}", "criticality": criticality,
                "criticality_value": 4 - criticality_levels.index(criticality),
                "shelf_life": shelf_life, "production_complexity": production_complexity,
                "substitutability": substitutability, "base_demand": base_demand,
                "base_production": base_production,
                "transportation_difficulty": np.random.uniform(0.1, 0.5)
            })
        return drugs

    def _sir_model(self, t, y, beta, gamma):
        """SIR model differential equations."""
        S, I, R = y
        N = S + I + R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def wave_severity_factor(self, wave_number):
        """Helper function to determine wave severity based on wave number."""
        if wave_number == 0:
            return np.random.uniform(0.05, 0.15)  # First wave severity
        else:
            return np.random.uniform(0.02, 0.08)  # Subsequent waves

    def _simulate_sir_with_waves(self, region, num_waves=None):
        """Simulate SIR model with multiple infection waves."""
        population = region["population"]
        
        # Determine number of waves based on region type and randomness
        if num_waves is None:
            region_type = region["type"]
            if region_type == "Urban":
                num_waves = np.random.randint(2, 4)  # Urban areas more likely to have multiple waves
            else:
                num_waves = np.random.randint(1, 3)  # Other areas might have fewer waves
        
        # Adjust severity based on the pandemic_severity parameter
        region_severity = self.pandemic_severity * np.random.uniform(0.7, 1.3)
        
        # Initialize arrays for tracking total cases over time
        total_cases = np.zeros(self.scenario_length)
        active_cases = np.zeros(self.scenario_length)
        new_cases_per_day = np.zeros(self.scenario_length)
        
        for wave in range(num_waves):
            # Randomly determine wave parameters
            wave_start = np.random.randint(0, max(1, self.scenario_length - 60))
            
            # Calculate remaining susceptible population for this wave
            # For subsequent waves, only a portion of the previously uninfected are susceptible
            if wave > 0:
                remaining_population = population - total_cases[wave_start]
                initial_susceptible = remaining_population * np.random.uniform(0.7, 0.95)
            else:
                initial_susceptible = population * np.random.uniform(0.9, 0.99)  # Almost everyone starts susceptible
            
            # Set SIR parameters for this wave
            # R0 typically between 1.5 and 3.5 for respiratory diseases
            r0 = np.random.uniform(1.5, 3.5) * region_severity
            
            # Recovery rate (1/gamma is the infectious period in days)
            # Typical infectious period: 5-14 days
            infectious_period = np.random.uniform(5, 14)
            gamma = 1.0 / infectious_period
            
            # Calculate beta from R0 and gamma
            beta = r0 * gamma
            
            # Initial conditions [S, I, R]
            # Start with a small number of infections
            initial_infected = max(1, population * 0.0001 * self.wave_severity_factor(wave))
            initial_recovered = 0
            
            # Ensure initial values don't exceed population
            initial_susceptible = min(initial_susceptible, population - initial_infected - initial_recovered)
            
            y0 = [initial_susceptible, initial_infected, initial_recovered]
            
            # Time points for the wave (from wave start to end of simulation)
            t_span = [0, self.scenario_length - wave_start]
            t_eval = np.arange(0, self.scenario_length - wave_start)
            
            # Solve the SIR differential equations
            solution = solve_ivp(
                lambda t, y: self._sir_model(t, y, beta, gamma),
                t_span,
                y0,
                t_eval=t_eval,
                method='RK45'
            )
            
            # Extract the solution
            S = solution.y[0]
            I = solution.y[1]
            R = solution.y[2]
            
            # Add this wave's contribution to total cases
            for i in range(len(t_eval)):
                day_idx = wave_start + i
                if day_idx < self.scenario_length:
                    # Active cases are directly from the I compartment
                    active_cases[day_idx] += I[i]
                    
                    # Calculate new cases for this day by looking at the decrease in susceptible
                    if i > 0:
                        new_infections = max(0, S[i-1] - S[i])
                        new_cases_per_day[day_idx] += new_infections
                    elif i == 0 and initial_infected > 0:
                        # For the first day, use initial infected as new cases
                        new_cases_per_day[day_idx] += initial_infected
                        
                    # Update total cumulative cases
                    if day_idx == 0:
                        total_cases[day_idx] = new_cases_per_day[day_idx]
                    else:
                        total_cases[day_idx] = total_cases[day_idx-1] + new_cases_per_day[day_idx]
        
        # Return both active cases and cumulative cases for flexibility
        return {
            "active_cases": active_cases,
            "total_cases": total_cases,
            "new_cases_per_day": new_cases_per_day
        }



    def _generate_epidemic_curves(self) -> Dict[int, np.ndarray]:
        """Generate epidemic curves using SIR model for each region."""
        curves = {}
        
        for region in self.regions:
            region_id = region["id"]
            
            # Simulate the SIR model for this region
            sir_results = self._simulate_sir_with_waves(region)
            
            # Store the active cases as the epidemic curve
            # This represents cases requiring healthcare resources (and thus drugs)
            curves[region_id] = sir_results["active_cases"]
        
        # Double-check that all regions have curves
        for region in self.regions:
            if region["id"] not in curves or len(curves[region["id"]]) == 0:
                if self.console:
                    self.console.print(f"[yellow]Warning: Region {region['id']} missing valid curve. Creating empty curve.[/]")
                curves[region["id"]] = np.zeros(self.scenario_length)
                
        return curves

    def _generate_disruptions(self) -> List[Dict]:
        disruptions = []
        # Manufacturing disruptions (using code15 parameters)
        for day in range(self.scenario_length):
            for drug in self.drugs:
                disruption_prob = self.disruption_probability * drug["production_complexity"] / 20 # code15 change
                if np.random.random() < disruption_prob / 10: # Make rare
                    duration = np.random.randint(3, 15) # code15 change
                    severity = np.random.uniform(0.2, 0.6) # code15 change
                    disruptions.append({
                        "type": "manufacturing", "drug_id": drug["id"],
                        "start_day": day, "end_day": min(day + duration, self.scenario_length - 1),
                        "severity": severity, "description": f"Manufacturing disruption for {drug['name']}"
                    })
        # Transportation disruptions
        for day in range(self.scenario_length):
            for region in self.regions:
                disruption_prob = self.disruption_probability * (1 - region["transportation_reliability"])
                if np.random.random() < disruption_prob / 10: # Make rare
                    duration = np.random.randint(2, 14)
                    severity = np.random.uniform(0.4, 0.8)
                    disruptions.append({
                        "type": "transportation", "region_id": region["id"],
                        "start_day": day, "end_day": min(day + duration, self.scenario_length - 1),
                        "severity": severity, "description": f"Transportation disruption to {region['name']}"
                    })
        return disruptions

    def get_daily_drug_demand(self, day: int, region_id: int, drug_id: int) -> float:
        """Calculate drug demand based on active cases from SIR model."""
        if day >= self.scenario_length: return 0
        drug = self.drugs[drug_id]; region = self.regions[region_id]
        day_idx = min(day, len(self.epidemic_curves[region_id]) - 1)
        
        # Get active cases for this day
        active_cases = self.epidemic_curves[region_id][day_idx]
        
        # Calculate demand based on active cases and drug base demand
        daily_demand = active_cases * drug["base_demand"] / 1000
        
        # Add some randomness to the demand
        daily_demand *= np.random.uniform(0.9, 1.1)
        
        return max(0, daily_demand) # Ensure non-negative demand

    def get_manufacturing_capacity(self, day: int, drug_id: int) -> float:
        drug = self.drugs[drug_id]; base_capacity = drug["base_production"]
        active_disruptions = [d for d in self.disruptions if d["type"] == "manufacturing" and d["drug_id"] == drug_id and d["start_day"] <= day <= d["end_day"]]
        capacity_multiplier = 1.0
        for disruption in active_disruptions: capacity_multiplier *= (1 - disruption["severity"])
        # return base_capacity * 
        return max(0.0, base_capacity * capacity_multiplier)         # Ensure capacity is not negative





    def get_transportation_capacity(self, day: int, region_id: int) -> float:
        region = self.regions[region_id]; base_capacity = 1.0
        active_disruptions = [d for d in self.disruptions if d["type"] == "transportation" and d["region_id"] == region_id and d["start_day"] <= day <= d["end_day"]]
        capacity_multiplier = 1.0
        for disruption in active_disruptions: capacity_multiplier *= (1 - disruption["severity"])
        # Ensure capacity doesn't become zero or negative due to severity > 1
        effective_multiplier = max(0.01, capacity_multiplier) # Set a minimum capacity floor
        return base_capacity * effective_multiplier * region["transportation_reliability"]