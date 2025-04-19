"""
Pandemic supply chain environment simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class PandemicSupplyChainEnvironment:
    """Simulates a pandemic supply chain with manufacturers, distributors, and hospitals."""
    
    def __init__(
        self,
        scenario_generator,
        initial_manufacturer_inventory: float = 5000,
        initial_distributor_inventory: float = 2000,
        initial_hospital_inventory: float = 500,
        initial_warehouse_inventory: float = 0,
        blockchain_interface = None,
        use_blockchain: bool = False,
        console = None #
    ):
        self.scenario = scenario_generator
        self.num_regions = len(scenario_generator.regions)
        self.num_drugs = len(scenario_generator.drugs)
        self.scenario_length = scenario_generator.scenario_length
        self.current_day = 0
        self.console = console

        # Store blockchain info
        self.blockchain = blockchain_interface
        self.use_blockchain = use_blockchain

        # Initialize inventories using code15 values
        self.inventories = {}
        for drug_id in range(self.num_drugs):
            self.inventories[drug_id] = {0: initial_manufacturer_inventory} # Manufacturer
            for region_id in range(self.num_regions): # Distributors
                self.inventories[drug_id][region_id + 1] = initial_distributor_inventory
            for region_id in range(self.num_regions): # Hospitals
                self.inventories[drug_id][self.num_regions + 1 + region_id] = initial_hospital_inventory

        self.warehouse_inventories = {drug_id: initial_warehouse_inventory for drug_id in range(self.num_drugs)}

        # Pipelines
        self.pipelines = {}
        for from_id in range(2 * self.num_regions + 1):
            self.pipelines[from_id] = {}
            for to_id in range(2 * self.num_regions + 1):
                self.pipelines[from_id][to_id] = {drug_id: [] for drug_id in range(self.num_drugs)}

        # Metrics
        self.stockouts = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.unfulfilled_demand = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.total_demand = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.patient_impact = {r: 0 for r in range(self.num_regions)}

        # History
        self.demand_history = []
        self.order_history = []
        self.allocation_history = []
        self.stockout_history = []
        self.production_history = []
        self.warehouse_release_history = []
        self.inventory_history = {}
        self.warehouse_history = {}
        self.pending_allocations = {} # For batching

        # Config (will be set in run_simulation)
        self.verbose = False
        self.warehouse_release_delay = 0
        self.allocation_batch_frequency = 1
        
        # Initialize history for day 0
        self.inventory_history[0] = {drug_id: self.inventories[drug_id].copy() for drug_id in range(self.num_drugs)}
        self.warehouse_history[0] = {drug_id: self.warehouse_inventories[drug_id] for drug_id in range(self.num_drugs)}

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)
    
    def reset(self):
        """Reset the environment."""
        self.current_day = 0
        initial_manufacturer_inventory = 5000
        initial_distributor_inventory = 2000
        initial_hospital_inventory = 500
        initial_warehouse_inventory = 0

        self.inventories = {}
        for drug_id in range(self.num_drugs):
            self.inventories[drug_id] = {0: initial_manufacturer_inventory}
            for region_id in range(self.num_regions):
                self.inventories[drug_id][region_id + 1] = initial_distributor_inventory
            for region_id in range(self.num_regions):
                self.inventories[drug_id][self.num_regions + 1 + region_id] = initial_hospital_inventory
        self.warehouse_inventories = {drug_id: initial_warehouse_inventory for drug_id in range(self.num_drugs)}

        self.pipelines = {}
        for from_id in range(2 * self.num_regions + 1):
            self.pipelines[from_id] = {}
            for to_id in range(2 * self.num_regions + 1):
                self.pipelines[from_id][to_id] = {drug_id: [] for drug_id in range(self.num_drugs)}

        self.stockouts = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.unfulfilled_demand = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.total_demand = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.patient_impact = {r: 0 for r in range(self.num_regions)}

        self.demand_history = []
        self.order_history = []
        self.allocation_history = []
        self.stockout_history = []
        self.production_history = []
        self.warehouse_release_history = []
        self.inventory_history = {}
        self.warehouse_history = {}
        self.pending_allocations = {}

        # Initialize history for day 0
        self.inventory_history[0] = {drug_id: self.inventories[drug_id].copy() for drug_id in range(self.num_drugs)}
        self.warehouse_history[0] = {drug_id: self.warehouse_inventories[drug_id] for drug_id in range(self.num_drugs)}

        return self.get_observations()

    def _process_production(self, production_actions: Dict):
        """Process drug production -> warehouse."""
        for drug_id, amount in production_actions.items():
            try:
                drug_id = int(drug_id)
                if not (0 <= drug_id < self.num_drugs): continue
                capacity = self.scenario.get_manufacturing_capacity(self.current_day, drug_id)
                actual_production = min(float(amount), capacity)
                if actual_production < 0: actual_production = 0 # Ensure non-negative
                if drug_id not in self.warehouse_inventories: self.warehouse_inventories[drug_id] = 0
                self.warehouse_inventories[drug_id] += actual_production
                self.production_history.append({
                    "day": self.current_day, "drug_id": drug_id, "amount": actual_production,
                    "released": False, "release_day": None
                })
                if self.verbose and actual_production > 0:
                    drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                    self._print(f"[blue]Production: {actual_production:.2f} units of {drug_name} -> warehouse[/]")
            except (ValueError, TypeError) as e:
                 self._print(f"[yellow]Error processing production for drug_id {drug_id}: {e}[/]")

    def _process_warehouse_release(self, release_delay: int):
        """Release inventory warehouse -> manufacturer after delay."""
        eligible = [e for e in self.production_history if e["day"] <= self.current_day - release_delay and not e.get("released", False)]
        released_count = 0
        for entry in eligible:
            drug_id = entry["drug_id"]; amount = entry["amount"]
            available = min(amount, self.warehouse_inventories.get(drug_id, 0))
            if available > 0:
                self.warehouse_inventories[drug_id] -= available
                if drug_id not in self.inventories: self.inventories[drug_id] = {} # Ensure drug key exists
                if 0 not in self.inventories[drug_id]: self.inventories[drug_id][0] = 0 # Ensure manu key exists
                self.inventories[drug_id][0] += available
                entry["released"] = True; entry["release_day"] = self.current_day
                self.warehouse_release_history.append({
                    "day": self.current_day, "drug_id": drug_id, "amount": available,
                    "production_day": entry["day"], "delay_days": self.current_day - entry["day"]
                })
                released_count += 1
                if self.verbose:
                     drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                     self._print(f"[cyan]Warehouse release: {available:.2f} units of {drug_name} released after {self.current_day - entry['day']} days[/]")

    def _process_batch_allocation(self, allocation_actions: Dict, batch_frequency: int):
        """Process batch allocation manufacturer -> distributors."""
        # Only process on batch days if frequency > 1
        if batch_frequency > 1 and self.current_day % batch_frequency != 0:
            # Store pending allocations
            for drug_id, region_allocs in allocation_actions.items():
                drug_id_str = str(drug_id)
                if drug_id_str not in self.pending_allocations: self.pending_allocations[drug_id_str] = {}
                for region_id, amount in region_allocs.items():
                    region_id_str = str(region_id)
                    current = self.pending_allocations[drug_id_str].get(region_id_str, 0)
                    self.pending_allocations[drug_id_str][region_id_str] = current + float(amount)
            return # Skip actual allocation

        # On batch day (or if frequency is 1), process combined allocations
        combined_allocations = {}
        # Add current actions
        for drug_id, region_allocs in allocation_actions.items():
            drug_id_str = str(drug_id)
            if drug_id_str not in combined_allocations: combined_allocations[drug_id_str] = {}
            for region_id, amount in region_allocs.items():
                region_id_str = str(region_id)
                combined_allocations[drug_id_str][region_id_str] = float(amount)

        # Merge with pending
        if hasattr(self, 'pending_allocations') and self.pending_allocations:
            if self.verbose and batch_frequency > 1:
                self._print(f"[green]Allocation batching: Processing batch on day {self.current_day}[/]")
            for drug_id_str, region_allocs in self.pending_allocations.items():
                if drug_id_str not in combined_allocations: combined_allocations[drug_id_str] = {}
                for region_id_str, amount in region_allocs.items():
                    current = combined_allocations[drug_id_str].get(region_id_str, 0)
                    combined_allocations[drug_id_str][region_id_str] = current + amount
            self.pending_allocations = {} # Clear pending

        # Convert keys back to integers for processing
        integer_allocations = {}
        for drug_id_str, region_allocs in combined_allocations.items():
            try:
                drug_id = int(drug_id_str)
                integer_allocations[drug_id] = {}
                for region_id_str, amount in region_allocs.items():
                    try:
                        region_id = int(region_id_str)
                        integer_allocations[drug_id][region_id] = amount
                    except ValueError: continue
            except ValueError: continue

        self._process_manufacturer_allocation(integer_allocations)

    def _process_manufacturer_allocation(self, allocation_actions: Dict):
        """Actual allocation manufacturer -> distributors."""
        for drug_id, region_allocations in allocation_actions.items():
            try:
                drug_id = int(drug_id)
                if drug_id not in self.inventories: self.inventories[drug_id] = {0: 0} # Ensure drug exists
                if 0 not in self.inventories[drug_id]: self.inventories[drug_id][0] = 0 # Ensure manu inv exists
                available_inventory = self.inventories[drug_id][0]
                total_allocated_for_drug = 0

                # Prepare requests for fair allocation tool
                requested_amounts = {}
                for region_id, amount in region_allocations.items():
                    try:
                        # Only consider positive requests
                        req_amount = float(amount)
                        if req_amount > 0:
                            requested_amounts[int(region_id)] = req_amount
                    except (ValueError, TypeError): continue

                if not requested_amounts or available_inventory <= 0: continue # Skip if no valid requests or no inventory

                # Use code15's fair allocation logic
                fair_allocations = self._calculate_fair_allocation(
                    drug_id, requested_amounts, available_inventory
                )

                # Process the calculated fair allocations
                for region_id, amount_to_allocate in fair_allocations.items():
                    if amount_to_allocate <= 0: continue # Skip zero allocations
                    region_id = int(region_id) # Ensure integer
                    distributor_id = region_id + 1

                    # Check inventory again before allocating this specific amount
                    current_manu_inv = self.inventories[drug_id][0]
                    actual_allocation = min(amount_to_allocate, current_manu_inv)

                    if actual_allocation <= 0: continue

                    base_lead_time = 1 + np.random.poisson(1) # 1-3 days
                    transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                    # Ensure factor is not zero to avoid division error
                    transport_capacity_factor = max(0.1, transport_capacity_factor)
                    adjusted_lead_time = max(1, int(round(base_lead_time / transport_capacity_factor)))
                    arrival_day = self.current_day + adjusted_lead_time

                    # Ensure pipeline structure exists
                    if 0 not in self.pipelines: self.pipelines[0] = {}
                    if distributor_id not in self.pipelines[0]: self.pipelines[0][distributor_id] = {}
                    if drug_id not in self.pipelines[0][distributor_id]: self.pipelines[0][distributor_id][drug_id] = []

                    self.pipelines[0][distributor_id][drug_id].append((actual_allocation, arrival_day))
                    self.inventories[drug_id][0] -= actual_allocation
                    total_allocated_for_drug += actual_allocation

                    self.allocation_history.append({
                        "day": self.current_day, "drug_id": drug_id, "from_id": 0,
                        "to_id": distributor_id, "amount": actual_allocation, "arrival_day": arrival_day
                    })

                    # ADD BLOCKCHAIN CALL for Manufacturer Allocation
                    if self.use_blockchain and self.blockchain:
                        try:
                            # Make sure region_id is int
                            bc_region_id = int(region_id)
                            bc_amount = int(actual_allocation) # Contract might expect integer
                            result = self.blockchain.record_allocation(drug_id, bc_region_id, bc_amount)
                            if result['status'] != 'success':
                                self._print(f"[yellow]BC Tx Failed (Manuf Alloc): {result.get('error', 'Unknown error')}[/]")
                        except Exception as e:
                            self._print(f"[yellow]BC Error (Manuf Alloc D{drug_id} R{region_id}): {e}[/]")

            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing manufacturer allocation for drug_id {drug_id}: {e}[/]")
                continue

    def _process_distributor_orders(self, distributor_orders: Dict):
        """Record orders distributor -> manufacturer."""
        for region_id, drug_orders in distributor_orders.items():
            try:
                region_id = int(region_id)
                distributor_id = region_id + 1
                for drug_id, amount in drug_orders.items():
                    try:
                         drug_id = int(drug_id)
                         amount = float(amount)
                         if amount > 0:
                             self.order_history.append({
                                 "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                                 "to_id": 0, "amount": amount
                             })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_distributor_allocation(self, allocation_actions: Dict):
        """Process allocation distributor -> hospital."""
        for region_id, drug_allocations in allocation_actions.items():
             try:
                region_id = int(region_id)
                distributor_id = region_id + 1
                hospital_id = self.num_regions + 1 + region_id
                for drug_id, amount in drug_allocations.items():
                     try:
                        drug_id = int(drug_id)
                        amount = float(amount)
                        if amount <= 0: continue

                        # Ensure inventory keys exist
                        if drug_id not in self.inventories: self.inventories[drug_id] = {}
                        if distributor_id not in self.inventories[drug_id]: self.inventories[drug_id][distributor_id] = 0

                        available_inventory = self.inventories[drug_id].get(distributor_id, 0)
                        actual_allocation = min(amount, available_inventory)

                        if actual_allocation <= 0: continue

                        base_lead_time = 1 # Typically short
                        transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                        transport_capacity_factor = max(0.1, transport_capacity_factor)
                        adjusted_lead_time = max(1, int(round(base_lead_time / transport_capacity_factor)))
                        arrival_day = self.current_day + adjusted_lead_time

                        # Ensure pipeline structure exists
                        if distributor_id not in self.pipelines: self.pipelines[distributor_id] = {}
                        if hospital_id not in self.pipelines[distributor_id]: self.pipelines[distributor_id][hospital_id] = {}
                        if drug_id not in self.pipelines[distributor_id][hospital_id]: self.pipelines[distributor_id][hospital_id][drug_id] = []

                        self.pipelines[distributor_id][hospital_id][drug_id].append((actual_allocation, arrival_day))
                        self.inventories[drug_id][distributor_id] -= actual_allocation

                        self.allocation_history.append({
                            "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                            "to_id": hospital_id, "amount": actual_allocation, "arrival_day": arrival_day
                        })
                     except (ValueError, TypeError, KeyError) as e:
                         self._print(f"[yellow]Error in distributor allocation for region {region_id}, drug {drug_id}: {e}[/]")
                         continue
             except (ValueError, TypeError): continue

    def _process_hospital_orders(self, hospital_orders: Dict):
        """Record orders hospital -> distributor."""
        for region_id, drug_orders in hospital_orders.items():
            try:
                region_id = int(region_id)
                hospital_id = self.num_regions + 1 + region_id
                distributor_id = region_id + 1
                for drug_id, amount in drug_orders.items():
                    try:
                        drug_id = int(drug_id)
                        amount = float(amount)
                        if amount > 0:
                            self.order_history.append({
                                "day": self.current_day, "drug_id": drug_id, "from_id": hospital_id,
                                "to_id": distributor_id, "amount": amount
                            })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_deliveries(self):
        """Process arrivals from pipelines."""
        for from_id in self.pipelines:
            for to_id in self.pipelines[from_id]:
                for drug_id in self.pipelines[from_id][to_id]:
                    # Ensure drug_id key exists in inventories before processing
                    if drug_id not in self.inventories:
                         self.inventories[drug_id] = {}

                    arrived = [(amt, day) for amt, day in self.pipelines[from_id][to_id][drug_id] if day <= self.current_day]
                    remaining = [(amt, day) for amt, day in self.pipelines[from_id][to_id][drug_id] if day > self.current_day]

                    total_arrived_amount = sum(amt for amt, day in arrived)

                    if total_arrived_amount > 0:
                        if to_id not in self.inventories[drug_id]: self.inventories[drug_id][to_id] = 0
                        self.inventories[drug_id][to_id] += total_arrived_amount

                    self.pipelines[from_id][to_id][drug_id] = remaining

    def _process_patient_demand(self):
        """Process patient demand at hospitals."""
        for region_id in range(self.num_regions):
            hospital_id = self.num_regions + 1 + region_id
            for drug_id in range(self.num_drugs):
                 try:
                    demand = self.scenario.get_daily_drug_demand(self.current_day, region_id, drug_id)
                    if demand <= 0: continue # Skip if no demand

                    # Ensure inventory keys exist
                    if drug_id not in self.inventories: self.inventories[drug_id] = {}
                    if hospital_id not in self.inventories[drug_id]: self.inventories[drug_id][hospital_id] = 0

                    available = self.inventories[drug_id].get(hospital_id, 0)

                    self.total_demand[drug_id][region_id] += demand
                    self.demand_history.append({
                        "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                        "demand": demand, "available": available
                    })

                    fulfilled = min(demand, available)
                    unfulfilled = demand - fulfilled

                    if fulfilled > 0:
                         self.inventories[drug_id][hospital_id] -= fulfilled

                    if unfulfilled > 0:
                        self.stockouts[drug_id][region_id] += 1 # Count days with any stockout
                        self.unfulfilled_demand[drug_id][region_id] += unfulfilled
                        self.stockout_history.append({
                            "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                            "demand": demand, "unfulfilled": unfulfilled
                        })
                        drug_criticality = self.scenario.drugs[drug_id]["criticality_value"]
                        impact = unfulfilled * drug_criticality
                        self.patient_impact[region_id] += impact

                 except KeyError as e:
                      # This might happen if hospital_id was somehow not initialized, though reset should handle it.
                      self._print(f"[red]Inventory key error processing demand: {e}. Skipping demand for hospital {hospital_id}, drug {drug_id}.[/]")
                      continue
                 except Exception as e:
                      self._print(f"[red]Unexpected error processing demand for hospital {region_id}, drug {drug_id}: {e}[/]")
                      continue
            
            # ADD BLOCKCHAIN CALL for Case Data Update (once per region per day)
            # Get current cases for the day just processed
            day_idx = min(self.current_day - 1, len(self.scenario.epidemic_curves[region_id]) - 1) # Day index for the demand just processed
            if day_idx >= 0:
                current_cases = self.scenario.epidemic_curves[region_id][day_idx]
                if self.use_blockchain and self.blockchain:
                    try:
                        bc_region_id = int(region_id)
                        bc_cases = int(current_cases) # Contract likely expects integer
                        result = self.blockchain.update_case_data(bc_region_id, bc_cases)
                        if result['status'] != 'success':
                             self._print(f"[yellow]BC Tx Failed (Case Data): {result.get('error', 'Unknown error')}[/]")
                    except Exception as e:
                        self._print(f"[yellow]BC Error (Case Data R{region_id}): {e}[/]")

    def _calculate_rewards(self) -> Dict:
        """Calculate rewards (simple example)."""
        rewards = {
            "manufacturer": 0,
            "distributors": {r: 0 for r in range(self.num_regions)},
            "hospitals": {r: 0 for r in range(self.num_regions)}
        }
        total_unfulfilled = sum(sum(v.values()) for v in self.unfulfilled_demand.values())
        rewards["manufacturer"] = -0.01 * total_unfulfilled # Penalize overall system inefficiency

        for r in range(self.num_regions):
            region_unfulfilled = sum(self.unfulfilled_demand[d][r] for d in range(self.num_drugs))
            rewards["distributors"][r] = -0.02 * region_unfulfilled # Penalize regional unfulfillment
            rewards["hospitals"][r] = -0.1 * self.patient_impact[r] # Directly penalize patient impact

        return rewards

    def _calculate_fair_allocation(self, drug_id: int, requested_amounts: Dict[int, float], available_inventory: float) -> Dict[int, float]:
        """Calculate fair allocation using code15 logic."""
        total_requested = sum(requested_amounts.values())
        allocations = {}

        # If enough inventory, fulfill all valid requests
        if total_requested <= available_inventory:
            for region_id, amount in requested_amounts.items():
                 if amount > 0: # Only allocate positive requests
                      allocations[region_id] = amount
            return allocations

        # Try blockchain allocation strategy
        if self.use_blockchain and self.blockchain:
            try:
                # Convert keys/values if necessary for the contract (e.g., keys to str, values to int)
                str_key_requests = {str(k): int(v) for k, v in requested_amounts.items() if v > 0}
                bc_available = int(available_inventory)

                allocations = self.blockchain.calculate_optimal_allocation(
                    drug_id, str_key_requests, bc_available
                )
                # Convert keys back to int, values back to float
                int_key_float_allocations = {int(k): float(v) for k, v in allocations.items()}
                self._print(f"[cyan]Using blockchain allocation strategy for Drug-{drug_id}[/]")
                # Ensure total allocated doesn't exceed available due to potential contract rounding/logic
                final_alloc = {}
                current_available = available_inventory
                for r_id, amount in int_key_float_allocations.items():
                    alloc = min(amount, current_available)
                    final_alloc[r_id] = alloc
                    current_available -= alloc
                return final_alloc # Return early if blockchain strategy is used and successful
            except Exception as e:
                self._print(f"[yellow]Blockchain allocation strategy query failed for Drug-{drug_id}: {e}. Falling back to local logic.[/]")
                # Fall through to local calculation if blockchain fails

        # Not enough inventory, use code15's prioritized allocation
        drug_info = self.scenario.drugs[drug_id]
        drug_criticality = drug_info["criticality_value"]
        drug_criticality_label = drug_info["criticality"]
        current_day_idx = min(self.current_day, self.scenario_length - 1)
        region_cases = {r_id: self.scenario.epidemic_curves[r_id][current_day_idx]
                        for r_id in requested_amounts.keys() if r_id < len(self.scenario.epidemic_curves)} # Added check for valid region_id index

        # Tiered allocation for Critical drugs (code15 logic)
        is_critical_drug = (drug_criticality_label == "Critical")
        max_cases = max(region_cases.values()) if region_cases else 0
        extreme_case_load_threshold = 50000
        high_case_load_threshold_factor = 0.3 # Regions above 30% of max cases are high-demand
        high_demand_reserve_ratio = 0.8 # Reserve 80% for high-demand regions for critical drugs

        # Filter out zero requests before proceeding
        valid_requests = {r: a for r, a in requested_amounts.items() if a > 0}
        if not valid_requests:
             return {} # No valid requests to allocate

        # Recalculate total requested based on valid requests
        total_requested = sum(valid_requests.values())
        # If somehow total requested became zero or less after filtering
        if total_requested <= 0:
             return {}

        # Check if tiered allocation applies
        apply_tiered_allocation = is_critical_drug and max_cases > extreme_case_load_threshold

        if apply_tiered_allocation:
             high_demand_regions = {r for r, cases in region_cases.items() if r in valid_requests and cases > max_cases * high_case_load_threshold_factor}
             low_demand_regions = {r for r in valid_requests if r not in high_demand_regions}

             high_demand_inventory = available_inventory * high_demand_reserve_ratio
             low_demand_inventory = available_inventory * (1 - high_demand_reserve_ratio)

             # Allocate to high-demand regions first
             high_demand_req = {r: valid_requests[r] for r in high_demand_regions}
             high_demand_cases = {r: region_cases[r] for r in high_demand_regions}
             allocations.update(self._allocate_proportionally(drug_info, high_demand_req, high_demand_cases, high_demand_inventory))

             # Allocate remaining to low-demand regions
             low_demand_req = {r: valid_requests[r] for r in low_demand_regions}
             low_demand_cases = {r: region_cases[r] for r in low_demand_regions}
             allocations.update(self._allocate_proportionally(drug_info, low_demand_req, low_demand_cases, low_demand_inventory))

        else:
            # Standard proportional allocation for non-critical or non-extreme scenarios
            allocations = self._allocate_proportionally(drug_info, valid_requests, region_cases, available_inventory)

        # Final check: ensure total allocation doesn't exceed available inventory due to rounding etc.
        total_allocated = sum(allocations.values())
        if total_allocated > available_inventory:
             # Simple scaling down if over-allocated (should rarely happen with min checks)
             if total_allocated > 0:
                 scale_down = available_inventory / total_allocated
                 allocations = {r: a * scale_down for r, a in allocations.items()}
             else: # If somehow total allocated is zero or negative, return empty
                 allocations = {}


        # Ensure no negative allocations
        allocations = {r: max(0, a) for r, a in allocations.items()}

        return allocations

    def _allocate_proportionally(self, drug_info: Dict, requests: Dict[int, float], cases: Dict[int, float], inventory: float) -> Dict[int, float]:
        """Helper for proportional allocation based on requests and cases (code15 logic)."""
        if inventory <= 0 or not requests:
            return {r: 0 for r in requests} # Return zero allocation if no inventory or requests

        priorities = {}
        total_priority = 0
        drug_criticality = drug_info["criticality_value"]
        extreme_case_load_threshold = 50000

        for region_id, request in requests.items():
             case_load = cases.get(region_id, 0)
             # Use exponential weighting for extreme cases (code15 logic)
             if case_load > extreme_case_load_threshold:
                 # Enhanced weighting: Use a base factor + exponential term
                 # Example: base * (case_load / threshold) ^ exponent
                 # Adjust exponent (1.5 in code15) for sensitivity
                 case_factor = (case_load / extreme_case_load_threshold) ** 1.5
                 # Add a base multiplier to prevent zero factor for zero cases
                 # and ensure regions with zero cases still get *some* priority if they request
                 base_factor = 0.1
                 priority_factor = base_factor + case_factor
             elif case_load > 0:
                 # Logarithmic scaling for non-extreme positive cases
                 priority_factor = 0.1 + np.log1p(case_load / 1000) # Scale cases before log
             else:
                 priority_factor = 0.1 # Minimal factor for zero cases

             # Combine request amount, case factor, and drug criticality
             priority = request * priority_factor * drug_criticality
             priorities[region_id] = max(0, priority) # Ensure non-negative priority
             total_priority += priorities[region_id]

        allocations = {}
        if total_priority <= 0:
             # If total priority is zero (e.g., all requests were zero or factors were zero),
             # distribute available inventory equally among requesting regions.
             num_requesters = len(requests)
             if num_requesters > 0:
                 equal_share = inventory / num_requesters
                 for region_id, request_amount in requests.items():
                      # Allocate the equal share, but not more than requested
                      allocations[region_id] = min(request_amount, equal_share)
             # else: allocations remains empty {}
        else:
            # Allocate proportionally based on calculated priorities
            for region_id, priority in priorities.items():
                proportion = priority / total_priority
                allocated_amount = proportion * inventory
                # Ensure allocation does not exceed the original request for that region
                allocations[region_id] = min(requests[region_id], allocated_amount)

        # Redistribute remaining inventory (due to capping at requested amounts)
        allocated_sum = sum(allocations.values())
        remaining_inventory = inventory - allocated_sum

        if remaining_inventory > 1e-6: # Only redistribute if meaningful amount remains
             # Identify regions that did not receive their full request
             under_allocated_regions = {r: (requests[r] - allocations[r])
                                        for r in requests if requests[r] > allocations[r] and (requests[r] - allocations[r]) > 1e-6}

             if under_allocated_regions:
                 # Recalculate priorities only for under-allocated regions based on their remaining need
                 remaining_needs_priorities = {}
                 total_remaining_priority = 0
                 for region_id, remaining_need in under_allocated_regions.items():
                      # Use the original priority score for weighting the remaining need
                      original_priority = priorities.get(region_id, 0)
                      # Weight the remaining need by its original priority contribution
                      # Alternative: just use remaining need or re-calculate priority based on remaining need
                      need_priority = remaining_need * (original_priority / requests[region_id] if requests[region_id] > 0 else 0)
                      # Fallback: if original priority was zero, use remaining need directly
                      if need_priority <= 0:
                          need_priority = remaining_need

                      remaining_needs_priorities[region_id] = max(0, need_priority)
                      total_remaining_priority += remaining_needs_priorities[region_id]


                 if total_remaining_priority > 0:
                      for region_id, need_priority in remaining_needs_priorities.items():
                           additional_proportion = need_priority / total_remaining_priority
                           additional_allocation = additional_proportion * remaining_inventory
                           # Allocate the additional amount, ensuring it doesn't exceed the remaining need
                           max_additional = under_allocated_regions[region_id]
                           allocations[region_id] += min(additional_allocation, max_additional)


        # Final check for non-negativity
        return {r: max(0, a) for r, a in allocations.items()}

    def get_observations(self) -> Dict:
        """Get observations for all agents."""
        return {
            "manufacturer": self._get_manufacturer_observation(),
            "distributors": {r: self._get_distributor_observation(r) for r in range(self.num_regions)},
            "hospitals": {r: self._get_hospital_observation(r) for r in range(self.num_regions)}
        }

    def _get_manufacturer_observation(self) -> Dict:
        """Get observation for manufacturer agent (code15 structure)."""
        current_day = min(self.current_day, self.scenario_length - 1)
        obs = {
            "day": self.current_day,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0) for drug_id in range(self.num_drugs)},
            "warehouse_inventories": {str(drug_id): self.warehouse_inventories.get(drug_id, 0) for drug_id in range(self.num_drugs)},
            "production_capacity": {str(drug_id): self.scenario.get_manufacturing_capacity(current_day, drug_id) for drug_id in range(self.num_drugs)},
            "pipeline": {}, # Outgoing pipeline
            "recent_orders": [], # Incoming orders from distributors
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)}, # Use string keys
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if d["type"] == "manufacturing" and d["start_day"] <= current_day <= d["end_day"]],
            "pending_releases": [], # From warehouse
            "pending_allocations": getattr(self, 'pending_allocations', {}), # For batching
            "batch_allocation_frequency": getattr(self, 'allocation_batch_frequency', 1),
            "is_batch_day": self.current_day % getattr(self, 'allocation_batch_frequency', 1) == 0 if getattr(self, 'allocation_batch_frequency', 1) > 0 else True,
        }

        # Populate Epi Data
        for region_id in range(self.num_regions):
             if region_id < len(self.scenario.epidemic_curves):
                  curve = self.scenario.epidemic_curves[region_id]
                  current_idx = min(current_day, len(curve) - 1)
                  prev_idx = max(0, current_idx - 7)
                  if current_idx >= 0:
                       current_cases = curve[current_idx]
                       prev_cases = curve[prev_idx]
                       trend = current_cases - prev_cases if current_idx > 0 else 0
                       obs["epidemiological_data"][str(region_id)] = {
                           "current_cases": current_cases, "case_trend": trend
                       }
                  else: # Handle case where curve might be empty or day is invalid
                       obs["epidemiological_data"][str(region_id)] = {"current_cases": 0, "case_trend": 0}
             else: # Handle case where region_id might be out of bounds for curves
                   obs["epidemiological_data"][str(region_id)] = {"current_cases": 0, "case_trend": 0}


        # Populate Outgoing Pipeline (Manu -> Dist)
        for region_id in range(self.num_regions):
            dist_id = region_id + 1
            obs["pipeline"][str(dist_id)] = {} # Use string key for region/dist ID
            for drug_id in range(self.num_drugs):
                 # Check existence of keys before summing
                if 0 in self.pipelines and dist_id in self.pipelines.get(0, {}) and drug_id in self.pipelines.get(0, {}).get(dist_id, {}):
                    obs["pipeline"][str(dist_id)][str(drug_id)] = sum(amount for amount, _ in self.pipelines[0][dist_id][drug_id])
                else:
                    obs["pipeline"][str(dist_id)][str(drug_id)] = 0


        # Populate Recent Orders (Dist -> Manu)
        obs["recent_orders"] = [o for o in self.order_history if o["to_id"] == 0 and o["day"] > self.current_day - 10]

        # Populate Pending Releases (Warehouse -> Manu)
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        obs["pending_releases"] = [{
            "drug_id": str(entry["drug_id"]), "amount": entry["amount"], "production_day": entry["day"],
            "days_in_warehouse": self.current_day - entry["day"],
            "expected_release_day": entry["day"] + release_delay
        } for entry in self.production_history if not entry.get("released", False)]

         # Add next batch day info
        freq = obs["batch_allocation_frequency"]
        if freq > 0:
             obs["next_batch_day"] = (self.current_day // freq + 1) * freq
             obs["days_to_next_batch"] = obs["next_batch_day"] - self.current_day
        else: # Handle freq=0 case (no batching)
             obs["next_batch_day"] = self.current_day
             obs["days_to_next_batch"] = 0


        return obs

    def _get_distributor_observation(self, region_id: int) -> Dict:
        """Get observation for distributor agent."""
        distributor_id = region_id + 1
        hospital_id = self.num_regions + 1 + region_id
        current_day = min(self.current_day, self.scenario_length - 1)
        region_id_str = str(region_id)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(distributor_id, 0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {}, # Manu -> This Dist
            "outbound_pipeline": {}, # This Dist -> Hosp
            "recent_orders": [], # Incoming orders from hospital
            "recent_allocations": [], # Incoming allocations from manufacturer
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if d["type"] == "transportation" and d["region_id"] == region_id and d["start_day"] <= current_day <= d["end_day"]]
        }

         # Populate Epi Data
        projected_demand = {} # Add projected demand here too, useful for dist agent
        for drug_id in range(self.num_drugs):
            projected_demand[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)

        if region_id < len(self.scenario.epidemic_curves):
             curve = self.scenario.epidemic_curves[region_id]
             current_idx = min(current_day, len(curve) - 1)
             prev_idx = max(0, current_idx - 7)
             if current_idx >= 0:
                  current_cases = curve[current_idx]
                  prev_cases = curve[prev_idx]
                  trend = current_cases - prev_cases if current_idx > 0 else 0
                  obs["epidemiological_data"] = {"current_cases": current_cases, "case_trend": trend, "projected_demand": projected_demand}
             else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand}
        else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand}


        # Populate Inbound Pipeline (Manu -> This Dist)
        for drug_id in range(self.num_drugs):
            if 0 in self.pipelines and distributor_id in self.pipelines.get(0, {}) and drug_id in self.pipelines.get(0, {}).get(distributor_id, {}):
                 obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[0][distributor_id][drug_id])
            else: obs["inbound_pipeline"][str(drug_id)] = 0


        # Populate Outbound Pipeline (This Dist -> Hosp)
        for drug_id in range(self.num_drugs):
             if distributor_id in self.pipelines and hospital_id in self.pipelines.get(distributor_id, {}) and drug_id in self.pipelines.get(distributor_id, {}).get(hospital_id, {}):
                  obs["outbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[distributor_id][hospital_id][drug_id])
             else: obs["outbound_pipeline"][str(drug_id)] = 0

        # Populate Recent Orders (Hosp -> This Dist)
        obs["recent_orders"] = [o for o in self.order_history if o["to_id"] == distributor_id and o["day"] > self.current_day - 10]

        # Populate Recent Allocations (Manu -> This Dist)
        obs["recent_allocations"] = [a for a in self.allocation_history if a["to_id"] == distributor_id and a["day"] > self.current_day - 10]

        return obs

    def _get_hospital_observation(self, region_id: int) -> Dict:
        """Get observation for hospital agent."""
        hospital_id = self.num_regions + 1 + region_id
        distributor_id = region_id + 1
        current_day = min(self.current_day, self.scenario_length - 1)
        region_id_str = str(region_id)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(hospital_id, 0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {}, # Dist -> This Hosp
            "recent_allocations": [], # Incoming allocations from distributor
            "demand_history": [], # Demand experienced by this hospital
            "stockout_history": [], # Stockouts experienced by this hospital
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if d["type"] == "transportation" and d["region_id"] == region_id and d["start_day"] <= current_day <= d["end_day"]]
        }

        # Populate Epi Data & Projected Demand
        projected_demand = {}
        for drug_id in range(self.num_drugs):
            projected_demand[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)

        if region_id < len(self.scenario.epidemic_curves):
             curve = self.scenario.epidemic_curves[region_id]
             current_idx = min(current_day, len(curve) - 1)
             prev_idx = max(0, current_idx - 7)
             if current_idx >= 0:
                  current_cases = curve[current_idx]
                  prev_cases = curve[prev_idx]
                  trend = current_cases - prev_cases if current_idx > 0 else 0
                  obs["epidemiological_data"] = {"current_cases": current_cases, "case_trend": trend, "projected_demand": projected_demand}
             else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand}
        else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand}


        # Populate Inbound Pipeline (Dist -> This Hosp)
        for drug_id in range(self.num_drugs):
            if distributor_id in self.pipelines and hospital_id in self.pipelines.get(distributor_id, {}) and drug_id in self.pipelines.get(distributor_id, {}).get(hospital_id, {}):
                obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[distributor_id][hospital_id][drug_id])
            else: obs["inbound_pipeline"][str(drug_id)] = 0


        # Populate Recent Allocations (Dist -> This Hosp)
        obs["recent_allocations"] = [a for a in self.allocation_history if a["to_id"] == hospital_id and a["day"] > self.current_day - 10]

        # Populate Demand History (This Hosp)
        obs["demand_history"] = [d for d in self.demand_history if d["region_id"] == region_id and d["day"] > self.current_day - 10]

        # Populate Stockout History (This Hosp)
        obs["stockout_history"] = [s for s in self.stockout_history if s["region_id"] == region_id and s["day"] > self.current_day - 10]

        return obs

    def step(self, actions: Dict):
        """Execute one day simulation step (code15 version)."""

        # Process actions affecting inventory flow
        # 1. Production (always happens, goes to warehouse)
        self._process_production(actions.get("manufacturer_production", {}))

        # 2. Warehouse Release (based on delay)
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        self._process_warehouse_release(release_delay)

        # 3. Orders placed (recorded, don't change inventory yet)
        self._process_distributor_orders(actions.get("distributor_orders", {}))
        self._process_hospital_orders(actions.get("hospital_orders", {}))

        # 4. Allocations (processed based on batching, create pipeline entries)
        batch_frequency = getattr(self, 'allocation_batch_frequency', 1)
        self._process_batch_allocation(actions.get("manufacturer_allocation", {}), batch_frequency) # Handles Manu -> Dist
        self._process_distributor_allocation(actions.get("distributor_allocation", {})) # Handles Dist -> Hosp (no batching here)

        # 5. Deliveries (process arrivals from pipeline, update inventory)
        self._process_deliveries()

        # 6. Patient Demand (consumes hospital inventory)
        self._process_patient_demand()

        # Track history AFTER all state changes for the day
        self.current_day += 1 # Increment day AFTER processing actions and demand for the 'previous' day index
        self.inventory_history[self.current_day] = {
            drug_id: self.inventories.get(drug_id, {}).copy() for drug_id in range(self.num_drugs)
        }
        self.warehouse_history[self.current_day] = {
            drug_id: self.warehouse_inventories.get(drug_id, 0) for drug_id in range(self.num_drugs)
        }

        rewards = self._calculate_rewards()
        done = self.current_day >= self.scenario_length
        observations = self.get_observations()
        info = {
            "stockouts": {d: r.copy() for d, r in self.stockouts.items()}, # Return copy
            "unfulfilled_demand": {d: r.copy() for d, r in self.unfulfilled_demand.items()},
            "patient_impact": self.patient_impact.copy(),
            "current_day": self.current_day,
            "warehouse_inventory": self.warehouse_inventories.copy(),
            "manufacturer_inventory": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0) for drug_id in range(self.num_drugs)},
            "pending_allocations": getattr(self, 'pending_allocations', {})
        }

        return observations, rewards, done, info