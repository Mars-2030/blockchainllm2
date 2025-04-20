# --- START OF FILE src/environment/supply_chain.py ---

"""
Pandemic supply chain environment simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import json # For potential debug printing
from src.tools.allocation import allocation_priority_tool


# Assuming scenario_generator provides necessary methods like get_daily_drug_demand etc.
# Assuming BlockchainInterface has methods like record_allocation, update_case_data, calculate_optimal_allocation

class PandemicSupplyChainEnvironment:
    """Simulates a pandemic supply chain with manufacturers, distributors, and hospitals."""

    def __init__(
        self,
        scenario_generator,
        initial_manufacturer_inventory: float = 5000,
        initial_distributor_inventory: float = 2000,
        initial_hospital_inventory: float = 500,
        initial_warehouse_inventory: float = 0, # Warehouse starts empty
        blockchain_interface = None,
        use_blockchain: bool = False,
        console = None # Rich console instance
    ):
        self.scenario = scenario_generator
        self.num_regions = len(scenario_generator.regions)
        self.num_drugs = len(scenario_generator.drugs)
        self.scenario_length = scenario_generator.scenario_length
        self.current_day = 0
        self.console = console

        # Store blockchain info
        self.blockchain = blockchain_interface
        self.use_blockchain = use_blockchain and self.blockchain is not None # Only use if interface is valid

        # Initialize inventories
        self.inventories = {} # {drug_id: {node_id: quantity}}
        for drug_id in range(self.num_drugs):
            self.inventories[drug_id] = {0: initial_manufacturer_inventory} # Manufacturer node_id = 0
            for region_id in range(self.num_regions):
                dist_id = region_id + 1 # Distributor node_ids = 1 to num_regions
                hosp_id = self.num_regions + 1 + region_id # Hospital node_ids = num_regions+1 to 2*num_regions
                self.inventories[drug_id][dist_id] = initial_distributor_inventory
                self.inventories[drug_id][hosp_id] = initial_hospital_inventory

        # Warehouse inventory (separate from manufacturer's usable stock)
        self.warehouse_inventories = {drug_id: initial_warehouse_inventory for drug_id in range(self.num_drugs)}

        # Pipelines {from_id: {to_id: {drug_id: [(amount, arrival_day), ...]}}}
        self.pipelines = {}
        node_ids = list(range(2 * self.num_regions + 1)) # 0=Manu, 1..N=Dist, N+1..2N=Hosp
        for from_id in node_ids:
            self.pipelines[from_id] = {}
            for to_id in node_ids:
                 if from_id != to_id: # No pipeline to self
                    self.pipelines[from_id][to_id] = {drug_id: [] for drug_id in range(self.num_drugs)}

        # Metrics
        self.stockouts = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Days with stockout at hospital
        self.unfulfilled_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Units unfulfilled at hospital
        self.total_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Total units demanded at hospital
        self.patient_impact = {r: 0.0 for r in range(self.num_regions)} # Weighted score based on unfulfilled and criticality

        # History Tracking
        self.demand_history = []        # Records hospital demand events
        self.order_history = []         # Records orders placed between nodes
        self.allocation_history = []    # Records allocations shipped (pipeline entries)
        self.stockout_history = []      # Records hospital stockout events (unfulfilled > 0)
        self.production_history = []    # Records manufacturer production amounts and release status
        self.warehouse_release_history = [] # Records when items move WH -> Manu inventory
        self.inventory_history = {}     # Daily snapshot {day: {drug_id: {node_id: quantity}}}
        self.warehouse_history = {}     # Daily snapshot {day: {drug_id: quantity}}
        self.pending_allocations = {}   # For batching manufacturer allocations {drug_id_str: {region_id_str: amount}}

        # Configuration (set externally, defaults provided)
        self.verbose = False
        self.warehouse_release_delay = 0 # Default: No delay
        self.allocation_batch_frequency = 1 # Default: Daily allocation

        # Initialize history for day 0
        self._record_daily_history()

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)

    def _record_daily_history(self):
        """Records inventory snapshots for the current day."""
        # Deep copy inventories to avoid modifying history later
        current_inv_snapshot = {}
        for drug_id, nodes in self.inventories.items():
            current_inv_snapshot[drug_id] = nodes.copy()
        self.inventory_history[self.current_day] = current_inv_snapshot

        self.warehouse_history[self.current_day] = self.warehouse_inventories.copy()

    def reset(self):
        """Reset the environment to its initial state."""
        # Re-initialize using the same parameters as __init__
        scenario = self.scenario # Keep the same scenario generator
        initial_manufacturer_inventory = 5000
        initial_distributor_inventory = 2000
        initial_hospital_inventory = 500
        initial_warehouse_inventory = 0

        self.__init__(
            scenario_generator=scenario,
            initial_manufacturer_inventory=initial_manufacturer_inventory,
            initial_distributor_inventory=initial_distributor_inventory,
            initial_hospital_inventory=initial_hospital_inventory,
            initial_warehouse_inventory=initial_warehouse_inventory,
            blockchain_interface=self.blockchain, # Keep existing interface
            use_blockchain=self.use_blockchain,   # Keep existing flag
            console=self.console                  # Keep existing console
        )
        # Explicitly reset attributes that might be set outside init
        self.verbose = False
        self.warehouse_release_delay = 0
        self.allocation_batch_frequency = 1

        return self.get_observations() # Return initial observations

    def _process_production(self, production_actions: Dict):
        """Process drug production -> warehouse."""
        for drug_id, amount in production_actions.items():
            try:
                drug_id = int(drug_id)
                if not (0 <= drug_id < self.num_drugs): continue
                # Use scenario to get capacity, potentially affected by disruptions
                capacity = self.scenario.get_manufacturing_capacity(self.current_day, drug_id)
                actual_production = min(max(0.0, float(amount)), capacity) # Ensure non-negative and capped

                if actual_production > 0:
                    if drug_id not in self.warehouse_inventories: self.warehouse_inventories[drug_id] = 0.0
                    self.warehouse_inventories[drug_id] += actual_production
                    self.production_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": actual_production,
                        "released": False, "release_day": None
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        self._print(f"[blue]Production: {actual_production:.1f} units of {drug_name} -> warehouse[/]")
            except (ValueError, TypeError, KeyError) as e:
                 self._print(f"[yellow]Error processing production for drug_id {drug_id}: {e}[/]")

    def _process_warehouse_release(self):
        """Release inventory warehouse -> manufacturer after delay."""
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        if release_delay < 0: release_delay = 0 # Ensure non-negative delay

        eligible = [e for e in self.production_history if not e.get("released", False) and e["day"] <= self.current_day - release_delay]

        for entry in eligible:
            try:
                drug_id = entry["drug_id"]
                amount_produced = entry["amount"]
                # Check how much is actually available in warehouse (could be less if allocated directly somehow, though not current logic)
                available_in_warehouse = self.warehouse_inventories.get(drug_id, 0.0)
                amount_to_release = min(amount_produced, available_in_warehouse)

                if amount_to_release > 0:
                    self.warehouse_inventories[drug_id] -= amount_to_release
                    # Ensure manufacturer inventory exists
                    if drug_id not in self.inventories: self.inventories[drug_id] = {0: 0.0}
                    if 0 not in self.inventories[drug_id]: self.inventories[drug_id][0] = 0.0
                    self.inventories[drug_id][0] += amount_to_release

                    entry["released"] = True
                    entry["release_day"] = self.current_day
                    self.warehouse_release_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": amount_to_release,
                        "production_day": entry["day"], "delay_days": self.current_day - entry["day"]
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        delay_taken = self.current_day - entry['day']
                        self._print(f"[cyan]Warehouse release: {amount_to_release:.1f} units of {drug_name} released after {delay_taken} days[/]")
            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing warehouse release for entry {entry}: {e}[/]")

    def _process_batch_allocation(self, allocation_actions: Dict):
        """ Accumulates or processes manufacturer allocations based on frequency. """
        batch_frequency = getattr(self, 'allocation_batch_frequency', 1)
        is_batch_day = (self.current_day + 1) % batch_frequency == 0 # Process at END of batch day (e.g., day 7, 14...)

        # Store current day's intended allocations
        for drug_id, region_allocs in allocation_actions.items():
            try:
                drug_id_str = str(int(drug_id))
                if drug_id_str not in self.pending_allocations: self.pending_allocations[drug_id_str] = {}
                for region_id, amount in region_allocs.items():
                    region_id_str = str(int(region_id))
                    current = self.pending_allocations[drug_id_str].get(region_id_str, 0.0)
                    self.pending_allocations[drug_id_str][region_id_str] = current + max(0.0, float(amount))
            except (ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing/accumulating allocation action: {e}[/]")
                continue

        # If it's batch day (or frequency is 1), process the accumulated allocations
        if batch_frequency == 1 or is_batch_day:
            if self.verbose and batch_frequency > 1 and self.pending_allocations:
                self._print(f"[green]Allocation batching: Processing accumulated batch on day {self.current_day}[/]")

            # Convert keys back to integers for processing function
            integer_allocations = {}
            for drug_id_str, region_allocs in self.pending_allocations.items():
                try:
                    drug_id = int(drug_id_str)
                    integer_allocations[drug_id] = {}
                    for region_id_str, amount in region_allocs.items():
                        try:
                            region_id = int(region_id_str)
                            if amount > 0.01: # Only process meaningful amounts
                                integer_allocations[drug_id][region_id] = amount
                        except ValueError: continue
                    # Remove drug if no regions have allocations after conversion/filtering
                    if not integer_allocations[drug_id]:
                        del integer_allocations[drug_id]
                except ValueError: continue

            if integer_allocations: # Only call process if there's something to allocate
                 self._process_manufacturer_allocation(integer_allocations)

            self.pending_allocations = {} # Clear pending batch regardless
        elif self.verbose and batch_frequency > 1:
             self._print(f"[dim]Allocation batching: Accumulating allocations, next batch on day {(self.current_day // batch_frequency + 1) * batch_frequency}[/dim]")


    def _process_manufacturer_allocation(self, allocation_actions: Dict):
        """Process actual allocation manufacturer -> distributors (creates pipeline entries)."""
        # This function now receives the potentially batched and combined allocations
        for drug_id, region_allocations in allocation_actions.items():
            try:
                drug_id = int(drug_id)
                # Check manufacturer has inventory entry, default to 0 if not
                available_inventory = self.inventories.get(drug_id, {}).get(0, 0.0)

                if available_inventory <= 0: continue # Skip if no inventory

                # Prepare requests for fair allocation (these are the *intended* amounts from the agent)
                requested_amounts = {}
                for region_id, amount in region_allocations.items():
                    try:
                         req_amount = float(amount)
                         if req_amount > 0: requested_amounts[int(region_id)] = req_amount
                    except (ValueError, TypeError): continue

                if not requested_amounts: continue # Skip if no valid requests after filtering

                # Use the _calculate_fair_allocation method to determine actual amounts based on availability
                fair_allocations = self._calculate_fair_allocation(
                    drug_id, requested_amounts, available_inventory
                )

                # Process the calculated fair allocations
                for region_id, amount_to_allocate in fair_allocations.items():
                    if amount_to_allocate <= 1e-6: continue # Skip negligible allocations

                    region_id = int(region_id) # Ensure integer
                    distributor_id = region_id + 1 # Map region_id to distributor node_id

                    # Double-check inventory before deducting (should be safe due to fair_alloc capping)
                    current_manu_inv = self.inventories.get(drug_id, {}).get(0, 0.0)
                    actual_allocation = min(amount_to_allocate, current_manu_inv)

                    if actual_allocation <= 1e-6: continue

                    # Calculate lead time considering disruptions
                    base_lead_time = 1 + np.random.poisson(1) # e.g., 1-3 days average
                    transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                    adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor)))) # Avoid division by zero
                    arrival_day = self.current_day + adjusted_lead_time

                    # Add to pipeline
                    if 0 not in self.pipelines: self.pipelines[0] = {}
                    if distributor_id not in self.pipelines[0]: self.pipelines[0][distributor_id] = {}
                    if drug_id not in self.pipelines[0][distributor_id]: self.pipelines[0][distributor_id][drug_id] = []
                    self.pipelines[0][distributor_id][drug_id].append((actual_allocation, arrival_day))

                    # Deduct from manufacturer inventory
                    self.inventories[drug_id][0] -= actual_allocation

                    # Log allocation history
                    self.allocation_history.append({
                        "day": self.current_day, "drug_id": drug_id, "from_id": 0,
                        "to_id": distributor_id, "amount": actual_allocation, "arrival_day": arrival_day
                    })

                    # --- Blockchain Call for Manufacturer Allocation ---
                    if self.use_blockchain:
                        try:
                            result = self.blockchain.record_allocation(
                                drug_id=int(drug_id),
                                region_id=int(region_id), # Use region_id for contract consistency
                                amount=int(round(actual_allocation)) # Contract might expect integer
                            )
                            if result.get('status') != 'success':
                                self._print(f"[yellow]BC Tx Failed (Manuf Alloc D{drug_id} R{region_id}): {result.get('error', 'Unknown BC error')}[/]")
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
                         if amount > 0.01: # Only record meaningful orders
                             self.order_history.append({
                                 "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                                 "to_id": 0, "amount": amount # Order TO manufacturer (node 0)
                             })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_distributor_allocation(self, allocation_actions: Dict):
        """Process allocation distributor -> hospital (creates pipeline entries)."""
        # Note: Distributor allocates directly based on its decision, not batched here.
        for region_id, drug_allocations in allocation_actions.items():
             try:
                region_id = int(region_id)
                distributor_id = region_id + 1
                hospital_id = self.num_regions + 1 + region_id

                for drug_id, amount in drug_allocations.items():
                     try:
                        drug_id = int(drug_id)
                        amount = float(amount)
                        if amount <= 1e-6: continue # Skip negligible amounts

                        # Check distributor inventory
                        available_inventory = self.inventories.get(drug_id, {}).get(distributor_id, 0.0)
                        actual_allocation = min(amount, available_inventory)

                        if actual_allocation <= 1e-6: continue

                        # Calculate lead time
                        base_lead_time = 1 # Typically short Dist -> Hosp
                        transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                        adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor))))
                        arrival_day = self.current_day + adjusted_lead_time

                        # Add to pipeline
                        if distributor_id not in self.pipelines: self.pipelines[distributor_id] = {}
                        if hospital_id not in self.pipelines[distributor_id]: self.pipelines[distributor_id][hospital_id] = {}
                        if drug_id not in self.pipelines[distributor_id][hospital_id]: self.pipelines[distributor_id][hospital_id][drug_id] = []
                        self.pipelines[distributor_id][hospital_id][drug_id].append((actual_allocation, arrival_day))

                        # Deduct from distributor inventory
                        self.inventories[drug_id][distributor_id] -= actual_allocation

                        # Log allocation history
                        self.allocation_history.append({
                            "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                            "to_id": hospital_id, "amount": actual_allocation, "arrival_day": arrival_day
                        })
                        # No separate blockchain call here, assume Manu Alloc covers chain state if needed
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
                distributor_id = region_id + 1 # Corresponding distributor
                for drug_id, amount in drug_orders.items():
                    try:
                        drug_id = int(drug_id)
                        amount = float(amount)
                        if amount > 0.01: # Record meaningful orders
                            self.order_history.append({
                                "day": self.current_day, "drug_id": drug_id, "from_id": hospital_id,
                                "to_id": distributor_id, "amount": amount # Order TO distributor
                            })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_deliveries(self):
        """Process arrivals from pipelines, updating inventories."""
        for from_id in list(self.pipelines.keys()): # Iterate over copy of keys
            if from_id not in self.pipelines: continue # Check if key still exists
            for to_id in list(self.pipelines[from_id].keys()):
                if to_id not in self.pipelines.get(from_id, {}): continue
                for drug_id in list(self.pipelines[from_id][to_id].keys()):
                    if drug_id not in self.pipelines.get(from_id, {}).get(to_id, {}): continue

                    current_pipeline = self.pipelines[from_id][to_id][drug_id]
                    arrived = [(amt, day) for amt, day in current_pipeline if day <= self.current_day]
                    remaining = [(amt, day) for amt, day in current_pipeline if day > self.current_day]

                    total_arrived_amount = sum(amt for amt, day in arrived)

                    if total_arrived_amount > 1e-6:
                        # Ensure target inventory exists
                        if drug_id not in self.inventories: self.inventories[drug_id] = {}
                        if to_id not in self.inventories[drug_id]: self.inventories[drug_id][to_id] = 0.0
                        self.inventories[drug_id][to_id] += total_arrived_amount
                        # Optional: Log delivery
                        # if self.verbose: self._print(f"[dim]Delivery: {total_arrived_amount:.1f} of Drug {drug_id} arrived at Node {to_id} from Node {from_id}[/dim]")

                    # Update pipeline
                    self.pipelines[from_id][to_id][drug_id] = remaining

    def _process_patient_demand(self):
        """Process patient demand at hospitals, calculate stockouts and impact."""
        for region_id in range(self.num_regions):
            hospital_id = self.num_regions + 1 + region_id
            for drug_id in range(self.num_drugs):
                 try:
                    # Get demand using scenario generator
                    demand = self.scenario.get_daily_drug_demand(self.current_day, region_id, drug_id)
                    demand = max(0.0, float(demand)) # Ensure non-negative float

                    if demand <= 1e-6: continue # Skip if no demand

                    # Ensure inventory keys exist, default to 0
                    available = self.inventories.get(drug_id, {}).get(hospital_id, 0.0)

                    # Track total demand
                    self.total_demand[drug_id][region_id] += demand
                    self.demand_history.append({
                        "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                        "demand": demand, "available": available
                    })

                    fulfilled = min(demand, available)
                    unfulfilled = demand - fulfilled

                    # Update inventory
                    if fulfilled > 0:
                         self.inventories[drug_id][hospital_id] -= fulfilled

                    # Track stockouts and impact
                    if unfulfilled > 1e-6: # If meaningful unfulfilled demand
                        self.stockouts[drug_id][region_id] += 1 # Increment stockout day counter
                        self.unfulfilled_demand[drug_id][region_id] += unfulfilled
                        self.stockout_history.append({
                            "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                            "demand": demand, "unfulfilled": unfulfilled
                        })
                        # Calculate patient impact (weighted by criticality)
                        drug_criticality = self.scenario.drugs[drug_id].get("criticality_value", 1)
                        impact = unfulfilled * drug_criticality
                        self.patient_impact[region_id] += impact

                 except KeyError as e:
                      self._print(f"[red]Inventory key error processing demand: {e}. Hospital {hospital_id}, Drug {drug_id}.[/]")
                      continue
                 except Exception as e:
                      self._print(f"[red]Unexpected error processing demand for hospital {region_id}, drug {drug_id}: {e}[/]")
                      continue

            # --- Blockchain Call for Case Data Update (after processing demand for the region) ---
            if self.use_blockchain:
                 day_idx = min(self.current_day, len(self.scenario.epidemic_curves.get(region_id, [])) - 1)
                 if day_idx >= 0 and region_id in self.scenario.epidemic_curves:
                     try:
                         current_cases = self.scenario.epidemic_curves[region_id][day_idx]
                         result = self.blockchain.update_case_data(
                             region_id=int(region_id),
                             num_cases=int(round(current_cases)) # Contract likely expects integer
                         )
                         if result.get('status') != 'success':
                              self._print(f"[yellow]BC Tx Failed (Case Data R{region_id}): {result.get('error', 'Unknown BC error')}[/]")
                     except Exception as e:
                         self._print(f"[yellow]BC Error (Case Data R{region_id}): {e}[/]")


    def _calculate_rewards(self) -> Dict:
        """Calculate simple rewards (negative values indicating penalties)."""
        # This is a basic reward structure, can be made more sophisticated
        rewards = {
            "manufacturer": 0.0,
            "distributors": {r: 0.0 for r in range(self.num_regions)},
            "hospitals": {r: 0.0 for r in range(self.num_regions)}
        }
        # Penalty for overall system inefficiency (unfulfilled demand)
        total_unfulfilled_all = sum(sum(v.values()) for v in self.unfulfilled_demand.values())
        rewards["manufacturer"] -= 0.001 * total_unfulfilled_all # Smaller penalty factor

        # Penalties for regional performance
        for r in range(self.num_regions):
            region_unfulfilled = sum(self.unfulfilled_demand[d][r] for d in range(self.num_drugs))
            rewards["distributors"][r] -= 0.002 * region_unfulfilled # Penalize regional unfulfillment
            rewards["hospitals"][r] -= 0.01 * self.patient_impact[r] # Penalize patient impact directly

        # Optional: Add rewards for holding inventory (negative cost) or penalties for excess inventory
        # Optional: Rewards for high service levels

        return rewards

    def _calculate_fair_allocation(self, drug_id: int, requested_amounts: Dict[int, float], available_inventory: float) -> Dict[int, float]:
        """
        Calculate fair allocation based on requests, criticality, case load, and potentially blockchain strategy.
        Uses the logic from allocation_priority_tool.
        """
        if available_inventory <= 1e-6:
            return {r_id: 0.0 for r_id in requested_amounts}

        # --- Try Blockchain Strategy First (if enabled) ---
        if self.use_blockchain:
            try:
                # Prepare data for potential contract interaction (e.g., string keys, integer amounts)
                str_key_int_requests = {str(k): int(round(v)) for k, v in requested_amounts.items() if v > 0}
                bc_available = int(round(available_inventory))

                if str_key_int_requests and bc_available > 0: # Only call if valid input
                    bc_allocations = self.blockchain.calculate_optimal_allocation(
                        drug_id=int(drug_id),
                        requests=str_key_int_requests,
                        available_supply=bc_available
                    )
                    # Convert back and validate
                    if isinstance(bc_allocations, dict):
                         # Convert keys back to int, values to float
                         int_key_float_allocations = {int(k): max(0.0, float(v)) for k, v in bc_allocations.items()}

                         # Ensure total doesn't exceed available due to contract logic/rounding
                         total_bc_allocated = sum(int_key_float_allocations.values())
                         if total_bc_allocated > available_inventory * 1.001: # Allow tiny overshoot
                              if total_bc_allocated > 0:
                                  scale_down = available_inventory / total_bc_allocated
                                  final_alloc = {r_id: amount * scale_down for r_id, amount in int_key_float_allocations.items()}
                                  self._print(f"[cyan]Using blockchain allocation strategy for Drug-{drug_id} (scaled down).[/]")
                              else: final_alloc = int_key_float_allocations # Keep as is if total is zero
                         else:
                              final_alloc = int_key_float_allocations
                              self._print(f"[cyan]Using blockchain allocation strategy for Drug-{drug_id}.[/]")

                         # Ensure all original requesting regions have an entry (even if 0)
                         for r_id in requested_amounts:
                              if r_id not in final_alloc: final_alloc[r_id] = 0.0

                         return {r: max(0.0, a) for r,a in final_alloc.items()} # Return cleaned blockchain result
                    else:
                        self._print(f"[yellow]Blockchain allocation strategy returned unexpected type for Drug-{drug_id}: {type(bc_allocations)}. Falling back.[/]")
            except Exception as e:
                self._print(f"[yellow]Blockchain allocation strategy query failed for Drug-{drug_id}: {e}. Falling back to local logic.[/]")
                # Fall through to local calculation if blockchain fails


        # --- Local Fair Allocation (using allocation_priority_tool logic) ---
        drug_info = self.scenario.drugs[drug_id]
        current_day_idx = min(self.current_day, self.scenario_length - 1)
        region_cases = {r_id: self.scenario.epidemic_curves[r_id][current_day_idx]
                        for r_id in requested_amounts.keys() if r_id < len(self.scenario.epidemic_curves)}

        # Call the centralized tool logic for consistent proportional allocation
        return allocation_priority_tool(drug_info, requested_amounts, region_cases, available_inventory)


    # --- Observation Methods ---

    # Step 3 Action 3: Enhanced Manufacturer Observation
    def _get_manufacturer_observation(self) -> Dict:
        """Get observation for manufacturer agent, including downstream summaries."""
        current_day = min(self.current_day, self.scenario_length - 1)
        obs = {
            "day": self.current_day,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "warehouse_inventories": {str(drug_id): self.warehouse_inventories.get(drug_id, 0.0) for drug_id in range(self.num_drugs)},
            "production_capacity": {str(drug_id): self.scenario.get_manufacturing_capacity(current_day, drug_id) for drug_id in range(self.num_drugs)},
            "pipeline": {}, # Outgoing pipeline (Manu -> Dists)
            "recent_orders": [], # Incoming orders from distributors (Dist -> Manu)
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "epidemiological_data": {}, # Regional cases/trends
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "manufacturing" and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)],
            "pending_releases": [], # From warehouse -> Manu
            "pending_allocations": getattr(self, 'pending_allocations', {}),
            "batch_allocation_frequency": getattr(self, 'allocation_batch_frequency', 1),
            # Calculate is_batch_day based on *next* day if processing at start of day
            "is_batch_day": (self.current_day + 1) % max(1, getattr(self, 'allocation_batch_frequency', 1)) == 0,
            "days_to_next_batch": 0, # Calculated below
            # --- Downstream Summaries Added ---
            "downstream_inventory_summary": {str(drug_id): {"total_distributor": 0.0, "total_hospital": 0.0, "total_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_pipeline_summary": {str(drug_id): {"manu_to_dist": 0.0, "dist_to_hosp": 0.0, "total_inbound_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_projected_demand_summary": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)},
        }

        # Calculate days_to_next_batch
        freq = obs["batch_allocation_frequency"]
        if freq > 0:
            current_batch_cycle_start = (self.current_day // freq) * freq
            next_batch_day = current_batch_cycle_start + freq
            obs["days_to_next_batch"] = next_batch_day - self.current_day
            obs["next_batch_day"] = next_batch_day # Add for potential LLM use
        else: # Continuous allocation
            obs["days_to_next_batch"] = 0
            obs["next_batch_day"] = self.current_day

        # Populate Epi Data & Downstream Summaries
        for drug_id in range(self.num_drugs):
            drug_id_str = str(drug_id)
            total_dist_inv = 0.0
            total_hosp_inv = 0.0
            total_dist_pipeline = 0.0
            total_hosp_pipeline = 0.0
            total_proj_demand = 0.0

            for region_id in range(self.num_regions):
                region_id_str = str(region_id)
                dist_id = region_id + 1
                hosp_id = self.num_regions + 1 + region_id

                # Epi data for this region
                if region_id < len(self.scenario.epidemic_curves):
                    curve = self.scenario.epidemic_curves[region_id]
                    current_idx = min(current_day, len(curve) - 1)
                    prev_idx = max(0, current_idx - 7)
                    if current_idx >= 0:
                       current_cases = curve[current_idx]
                       prev_cases = curve[prev_idx] if current_idx >= 7 else curve[0]
                       trend = current_cases - prev_cases # Trend over last 7 days (or since start)
                       obs["epidemiological_data"][region_id_str] = {
                           "current_cases": current_cases, "case_trend": trend
                       }
                    else: obs["epidemiological_data"][region_id_str] = {"current_cases": 0, "case_trend": 0}
                else: obs["epidemiological_data"][region_id_str] = {"current_cases": 0, "case_trend": 0}

                # Downstream Inventories
                total_dist_inv += self.inventories.get(drug_id, {}).get(dist_id, 0.0)
                total_hosp_inv += self.inventories.get(drug_id, {}).get(hosp_id, 0.0)

                # Downstream Pipelines
                if 0 in self.pipelines and dist_id in self.pipelines.get(0, {}) and drug_id in self.pipelines.get(0, {}).get(dist_id, {}):
                    total_dist_pipeline += sum(amount for amount, _ in self.pipelines[0][dist_id][drug_id])
                if dist_id in self.pipelines and hosp_id in self.pipelines.get(dist_id, {}) and drug_id in self.pipelines.get(dist_id, {}).get(hosp_id, {}):
                     total_hosp_pipeline += sum(amount for amount, _ in self.pipelines[dist_id][hosp_id][drug_id])

                # Projected Demand (Sum across regions for this drug)
                total_proj_demand += self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)


            # Store summarized downstream info
            obs["downstream_inventory_summary"][drug_id_str]["total_distributor"] = total_dist_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_hospital"] = total_hosp_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_downstream"] = total_dist_inv + total_hosp_inv
            obs["downstream_pipeline_summary"][drug_id_str]["manu_to_dist"] = total_dist_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["dist_to_hosp"] = total_hosp_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["total_inbound_downstream"] = total_dist_pipeline + total_hosp_pipeline
            obs["downstream_projected_demand_summary"][drug_id_str] = total_proj_demand


        # Populate Outgoing Pipeline (Manu -> All Dists) - Simplified view for Manu
        obs["pipeline"]["total_to_distributors"] = {}
        for drug_id in range(self.num_drugs):
             total_outgoing = 0.0
             for region_id in range(self.num_regions):
                 dist_id = region_id + 1
                 if 0 in self.pipelines and dist_id in self.pipelines.get(0, {}) and drug_id in self.pipelines.get(0, {}).get(dist_id, {}):
                     total_outgoing += sum(amount for amount, _ in self.pipelines[0][dist_id][drug_id])
             obs["pipeline"]["total_to_distributors"][str(drug_id)] = total_outgoing


        # Populate Recent Orders (Dist -> Manu) - Last 7 days
        obs["recent_orders"] = [o for o in self.order_history if o["to_id"] == 0 and o["day"] > self.current_day - 7]

        # Populate Pending Releases (Warehouse -> Manu)
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        obs["pending_releases"] = [{
            "drug_id": str(entry["drug_id"]), "amount": entry["amount"], "production_day": entry["day"],
            "days_in_warehouse": self.current_day - entry["day"],
            "expected_release_day": entry["day"] + release_delay
        } for entry in self.production_history if not entry.get("released", False)]

        return obs

    # Step 1 Change: Ensure projected_demand is included
    def _get_distributor_observation(self, region_id: int) -> Dict:
        """Get observation for distributor agent."""
        distributor_id = region_id + 1
        hospital_id = self.num_regions + 1 + region_id
        current_day = min(self.current_day, self.scenario_length - 1)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(distributor_id, 0.0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # Manu -> This Dist
            "outbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # This Dist -> Hosp
            "recent_orders": [], # Incoming orders from hospital (Hosp -> This Dist)
            "recent_allocations": [], # Incoming allocations from manufacturer (Manu -> This Dist)
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            "epidemiological_data": {}, # Cases, trend, and *projected hospital demand*
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)]
        }

        # Populate Epi Data & Projected Demand for the hospital in this region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)

        if region_id < len(self.scenario.epidemic_curves):
             curve = self.scenario.epidemic_curves[region_id]
             current_idx = min(current_day, len(curve) - 1)
             prev_idx = max(0, current_idx - 7)
             if current_idx >= 0:
                  current_cases = curve[current_idx]
                  prev_cases = curve[prev_idx] if current_idx >= 7 else curve[0]
                  trend = current_cases - prev_cases
                  obs["epidemiological_data"] = {"current_cases": current_cases, "case_trend": trend, "projected_demand": projected_demand_this_region}
             else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand_this_region}
        else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand_this_region}


        # Populate Inbound Pipeline (Manu -> This Dist)
        for drug_id in range(self.num_drugs):
            if 0 in self.pipelines and distributor_id in self.pipelines.get(0, {}) and drug_id in self.pipelines.get(0, {}).get(distributor_id, {}):
                 obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[0][distributor_id][drug_id])

        # Populate Outbound Pipeline (This Dist -> Hosp)
        for drug_id in range(self.num_drugs):
             if distributor_id in self.pipelines and hospital_id in self.pipelines.get(distributor_id, {}) and drug_id in self.pipelines.get(distributor_id, {}).get(hospital_id, {}):
                  obs["outbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[distributor_id][hospital_id][drug_id])

        # Populate Recent Orders (Hosp -> This Dist) - Last 7 days
        obs["recent_orders"] = [o for o in self.order_history if o["to_id"] == distributor_id and o["day"] > self.current_day - 7]

        # Populate Recent Allocations (Manu -> This Dist) - Last 7 days
        obs["recent_allocations"] = [a for a in self.allocation_history if a["to_id"] == distributor_id and a["day"] > self.current_day - 7]

        return obs

    # Step 1 Change: Ensure projected_demand is included
    def _get_hospital_observation(self, region_id: int) -> Dict:
        """Get observation for hospital agent."""
        hospital_id = self.num_regions + 1 + region_id
        distributor_id = region_id + 1
        current_day = min(self.current_day, self.scenario_length - 1)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(hospital_id, 0.0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # Dist -> This Hosp
            "recent_allocations": [], # Incoming allocations from distributor (Dist -> This Hosp)
            "demand_history": [], # Demand experienced by this hospital
            "stockout_history": [], # Stockouts experienced by this hospital
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            "epidemiological_data": {}, # Cases, trend, and *its own* projected demand
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)]
        }

        # Populate Epi Data & Projected Demand for this hospital's region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)

        if region_id < len(self.scenario.epidemic_curves):
             curve = self.scenario.epidemic_curves[region_id]
             current_idx = min(current_day, len(curve) - 1)
             prev_idx = max(0, current_idx - 7)
             if current_idx >= 0:
                  current_cases = curve[current_idx]
                  prev_cases = curve[prev_idx] if current_idx >= 7 else curve[0]
                  trend = current_cases - prev_cases
                  obs["epidemiological_data"] = {"current_cases": current_cases, "case_trend": trend, "projected_demand": projected_demand_this_region}
             else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand_this_region}
        else: obs["epidemiological_data"] = {"current_cases": 0, "case_trend": 0, "projected_demand": projected_demand_this_region}


        # Populate Inbound Pipeline (Dist -> This Hosp)
        for drug_id in range(self.num_drugs):
            if distributor_id in self.pipelines and hospital_id in self.pipelines.get(distributor_id, {}) and drug_id in self.pipelines.get(distributor_id, {}).get(hospital_id, {}):
                obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines[distributor_id][hospital_id][drug_id])

        # Populate Recent Allocations (Dist -> This Hosp) - Last 7 days
        obs["recent_allocations"] = [a for a in self.allocation_history if a["to_id"] == hospital_id and a["day"] > self.current_day - 7]

        # Populate Demand History (This Hosp) - Last 7 days
        obs["demand_history"] = [d for d in self.demand_history if d.get("region_id") == region_id and d.get("day", -1) > self.current_day - 7]

        # Populate Stockout History (This Hosp) - Last 7 days
        obs["stockout_history"] = [s for s in self.stockout_history if s.get("region_id") == region_id and s.get("day", -1) > self.current_day - 7]

        return obs


    def get_observations(self) -> Dict:
        """Get observations for all agents."""
        # Ensure all necessary keys exist even if empty, especially for the first step
        manufacturer_obs = self._get_manufacturer_observation()
        distributor_obs = {r: self._get_distributor_observation(r) for r in range(self.num_regions)}
        hospital_obs = {r: self._get_hospital_observation(r) for r in range(self.num_regions)}

        return {
            "manufacturer": manufacturer_obs,
            "distributors": distributor_obs,
            "hospitals": hospital_obs
        }


    def step(self, actions: Dict):
        """Execute one day simulation step."""

        # 1. Production (to warehouse)
        self._process_production(actions.get("manufacturer_production", {}))

        # 2. Warehouse Release (to manufacturer inventory)
        self._process_warehouse_release() # Uses self.warehouse_release_delay

        # 3. Record Orders (does not move inventory)
        self._process_distributor_orders(actions.get("distributor_orders", {}))
        self._process_hospital_orders(actions.get("hospital_orders", {}))

        # 4. Process Allocations (moves inventory into pipelines)
        # Manufacturer allocation might be batched
        self._process_batch_allocation(actions.get("manufacturer_allocation", {}))
        # Distributor allocation happens daily (as per current logic)
        self._process_distributor_allocation(actions.get("distributor_allocation", {}))

        # 5. Process Deliveries (moves inventory from pipelines to destinations)
        self._process_deliveries()

        # 6. Process Patient Demand (consumes hospital inventory, calculates metrics)
        self._process_patient_demand()

        # 7. Increment Day and Record History AFTER all state changes
        self.current_day += 1
        self._record_daily_history() # Record inventory/warehouse state for the *end* of the day just processed

        # 8. Calculate Rewards and Check Termination
        rewards = self._calculate_rewards()
        done = self.current_day >= self.scenario_length
        observations = self.get_observations() # Get observations for the START of the *next* day

        # Info dictionary for logging/analysis
        info = {
            "stockouts": {d: r.copy() for d, r in self.stockouts.items()},
            "unfulfilled_demand": {d: r.copy() for d, r in self.unfulfilled_demand.items()},
            "patient_impact": self.patient_impact.copy(),
            "current_day": self.current_day, # Day number we are about to start
            "warehouse_inventory": self.warehouse_inventories.copy(),
            "manufacturer_inventory": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "pending_allocations": getattr(self, 'pending_allocations', {}) # Show pending for next batch
        }

        return observations, rewards, done, info

# --- END OF FILE src/environment/supply_chain.py ---