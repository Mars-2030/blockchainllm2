// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19; // Ensure this version matches your hardhat.config.js

import "@openzeppelin/contracts/access/Ownable.sol"; // For owner control

/**
 * @title SupplyChainData
 * @dev Stores trusted supply chain data (cases, drug info) and handles fair allocation logic.
 *      Intended for use in a simulation environment where an external process (like a Python script)
 *      updates state (cases) and triggers allocation based on simulated requests.
 */
contract SupplyChainData is Ownable {
    // --- State Variables ---

    // Mapping from region ID (uint256) to its latest reported case count (uint256)
    mapping(uint256 => uint256) public regionalCaseCounts;

    // Mapping from drug ID (uint256) to its criticality value (e.g., 1=Low, 4=Critical)
    // This needs to be set by the owner post-deployment for the allocation logic to work.
    mapping(uint256 => uint256) public drugCriticalities;

    // --- Events ---

    /**
     * @dev Emitted when a region's case count is updated by the owner.
     * @param regionId The ID of the region being updated.
     * @param newCaseCount The new case count recorded.
     * @param timestamp The block timestamp when the update occurred.
     */
    event RegionalCaseCountUpdated(uint256 indexed regionId, uint256 newCaseCount, uint256 timestamp);

    /**
     * @dev Emitted when the criticality for a drug is set or updated by the owner.
     * @param drugId The ID of the drug.
     * @param criticalityValue The new criticality value assigned.
     */
    event DrugCriticalitySet(uint256 indexed drugId, uint256 criticalityValue);

    /**
     * @dev Emitted when the fair allocation logic is executed.
     * @param drugId The ID of the drug that was allocated.
     * @param regionIds The array of region IDs that requested the drug.
     * @param allocatedAmounts The corresponding array of amounts allocated to each region.
     * @param totalAvailable The total inventory available for allocation at the time of execution.
     * @param timestamp The block timestamp when the allocation was executed.
     */
    event AllocationExecuted(
        uint256 indexed drugId,
        uint256[] regionIds,
        uint256[] allocatedAmounts,
        uint256 totalAvailable,
        uint256 timestamp
    );

    // --- Functions ---

    /**
     * @dev Contract constructor sets the initial owner.
     * @param initialOwner The address designated as the initial owner of the contract.
     */
    constructor(address initialOwner) Ownable(initialOwner) {}

    /**
     * @dev Updates the case count for a specific region.
     *      Only callable by the contract owner (e.g., a trusted oracle or simulation controller).
     * @param _regionId The ID of the region to update.
     * @param _cases The new case count for the region.
     */
    function updateRegionalCaseCount(uint256 _regionId, uint256 _cases) external onlyOwner {
        regionalCaseCounts[_regionId] = _cases;
        emit RegionalCaseCountUpdated(_regionId, _cases, block.timestamp);
    }

    /**
     * @dev Sets the criticality value for a specific drug.
     *      Needed for the fair allocation logic. Should be set during deployment/setup.
     *      Only callable by the contract owner.
     * @param _drugId The ID of the drug.
     * @param _criticalityValue The criticality score (e.g., 1-4). Must be greater than 0.
     */
    function setDrugCriticality(uint256 _drugId, uint256 _criticalityValue) external onlyOwner {
        require(_criticalityValue > 0, "Criticality must be positive");
        drugCriticalities[_drugId] = _criticalityValue;
        emit DrugCriticalitySet(_drugId, _criticalityValue);
    }

     /**
     * @dev Retrieves the latest stored case count for a specific region.
     * @param _regionId The ID of the region.
     * @return The case count stored on the blockchain.
     */
    function getRegionalCaseCount(uint256 _regionId) external view returns (uint256) {
        return regionalCaseCounts[_regionId];
    }

     /**
     * @dev Retrieves the stored criticality value for a specific drug.
     * @param _drugId The ID of the drug.
     * @return The criticality value stored on the blockchain. Returns 0 if not set.
     */
    function getDrugCriticality(uint256 _drugId) external view returns (uint256) {
        return drugCriticalities[_drugId];
    }

    /**
     * @dev Executes fair allocation based on requests, available inventory, drug criticality, and stored case counts.
     *      Callable by any address (e.g., the manufacturer's backend/simulation environment).
     *      NOTE: This uses a simplified priority calculation suitable for Solidity (request * criticality * cases).
     *      Amounts (_requestedAmounts, _availableInventory) are expected as integers.
     *      It's assumed that the calling application handles conversion from/to decimal representations if needed.
     *
     * @param _drugId ID of the drug being allocated.
     * @param _regionIds Array of region IDs requesting the drug.
     * @param _requestedAmounts Array of corresponding requested amounts (as integers). Must match length of _regionIds.
     * @param _availableInventory Total available inventory for this drug (as integer). Must be positive.
     * @return allocatedAmounts Array of allocated amounts (as integers) corresponding to _regionIds.
     */
    function executeFairAllocation(
        uint256 _drugId,
        uint256[] memory _regionIds,
        uint256[] memory _requestedAmounts,
        uint256 _availableInventory
    ) external returns (uint256[] memory) {
        // --- Input Validation ---
        require(_regionIds.length == _requestedAmounts.length, "Input arrays must have same length");
        require(_availableInventory > 0, "No inventory available");

        uint256 numRegions = _regionIds.length;
        uint256[] memory allocatedAmounts = new uint256[](numRegions); // Initialize allocation array
        uint256 totalRequested = 0;

        // --- Calculate Total Requested Amount ---
        for (uint i = 0; i < numRegions; i++) {
            // Ensure individual requests are non-negative (though logic handles zero later)
            // require(_requestedAmounts[i] >= 0, "Requested amount cannot be negative"); // Optional check
            totalRequested += _requestedAmounts[i];
        }

        // --- Case 1: Enough Inventory ---
        // If total requested is less than or equal to available, fulfill all requests.
        if (totalRequested <= _availableInventory) {
            for (uint i = 0; i < numRegions; i++) {
                allocatedAmounts[i] = _requestedAmounts[i];
            }
            emit AllocationExecuted(_drugId, _regionIds, allocatedAmounts, _availableInventory, block.timestamp);
            return allocatedAmounts; // Return fulfilled amounts
        }

        // --- Case 2: Not Enough Inventory - Calculate Priorities ---
        uint256 drugCriticality = drugCriticalities[_drugId];
        // Criticality must be set via setDrugCriticality before this can work correctly.
        require(drugCriticality > 0, "Drug criticality not set");

        uint256[] memory priorities = new uint256[](numRegions);
        uint256 totalPriority = 0;

        // Calculate priority for each region's request
        for (uint i = 0; i < numRegions; i++) {
            if (_requestedAmounts[i] > 0) { // Only consider positive requests for priority
                // Fetch case count from blockchain storage
                uint256 regionCases = regionalCaseCounts[_regionIds[i]];
                // Simplified priority score: request * criticality * (cases + 1)
                // Adding 1 to cases avoids zero priority when cases are 0 but request is positive.
                uint256 priority = _requestedAmounts[i] * drugCriticality * (regionCases + 1);
                priorities[i] = priority;
                totalPriority += priority;
            } else {
                priorities[i] = 0; // Zero priority for zero requests
            }
        }

        uint256 actuallyAllocatedTotal = 0; // Track sum of allocations for potential scaling

        // --- Allocate Based on Priority ---
        if (totalPriority > 0) {
            // Proportional allocation: amount = available * (region_priority / total_priority)
            // Multiply first to maintain precision with integer division.
            for (uint i = 0; i < numRegions; i++) {
                if (priorities[i] > 0) {
                    // Calculate the region's proportional share of the available inventory
                    uint256 proportionalShare = (_availableInventory * priorities[i]) / totalPriority;
                    // Allocate the minimum of the calculated share and the original request
                    allocatedAmounts[i] = min(proportionalShare, _requestedAmounts[i]);
                    actuallyAllocatedTotal += allocatedAmounts[i];
                } else {
                     allocatedAmounts[i] = 0; // No allocation if priority was zero
                }
            }

            // --- Sanity Check & Correction (Optional but Recommended) ---
            // Check if rounding errors caused overallocation (unlikely with uint but possible with complex math).
            if (actuallyAllocatedTotal > _availableInventory) {
                 // This indicates a potential math error or overflow, should ideally not happen.
                 // Fallback: Scale down allocations proportionally to match available inventory.
                 // Using 1 ether (1e18) helps maintain precision during scaling calculation.
                 uint256 scaleFactor = (_availableInventory * 1 ether) / actuallyAllocatedTotal;
                 for (uint i = 0; i < numRegions; i++) {
                      allocatedAmounts[i] = (allocatedAmounts[i] * scaleFactor) / 1 ether;
                 }
                 // Recalculate total to ensure it's correct after scaling (for event emission)
                 actuallyAllocatedTotal = _availableInventory; // Should now exactly match
            }
             // Note: Integer division might lead to *under* allocation (remainder not distributed).
             // For this simulation, we accept that minor amounts might remain unallocated in scarce scenarios.
             // A more complex implementation could redistribute the remainder.

        } else {
            // Fallback: If total priority is 0 (e.g., zero cases, zero requests, or zero criticality),
            // distribute available inventory equally among positive requesters, capped by their request.
             uint positiveRequesters = 0;
             for(uint i = 0; i < numRegions; i++){
                 if(_requestedAmounts[i] > 0) {
                     positiveRequesters++;
                 }
             }
             if (positiveRequesters > 0) {
                 uint equalShare = _availableInventory / positiveRequesters;
                 for(uint i = 0; i < numRegions; i++){
                      if(_requestedAmounts[i] > 0) {
                           allocatedAmounts[i] = min(_requestedAmounts[i], equalShare);
                      } else {
                           allocatedAmounts[i] = 0;
                      }
                 }
             }
             // If no positive requests, allocatedAmounts remains all zeros.
        }

        // Emit event with the final calculated allocations
        emit AllocationExecuted(_drugId, _regionIds, allocatedAmounts, _availableInventory, block.timestamp);
        return allocatedAmounts;
    }

    /**
     * @dev Internal helper function to find the minimum of two unsigned integers.
     *      Using internal pure avoids external calls and saves gas.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}