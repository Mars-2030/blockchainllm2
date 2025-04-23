const { ethers } = require("hardhat");
require("dotenv").config();
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  const deployerAddress = await deployer.getAddress();
  console.log("Deploying contracts with the account:", deployerAddress);

  const balance = await ethers.provider.getBalance(deployerAddress);
  console.log("Account balance:", ethers.formatEther(balance));

  // Deploy SupplyChainData contract
  const supplyChainDataFactory = await ethers.getContractFactory("SupplyChainData");
  // Pass deployer address as initial owner
  const supplyChainData = await supplyChainDataFactory.deploy(deployerAddress);
  await supplyChainData.waitForDeployment(); // Wait for deployment confirmation
  const contractAddress = await supplyChainData.getAddress();


  console.log(`SupplyChainData contract deployed to: ${contractAddress}`);

  // --- Post-Deployment Setup ---
  console.log("Setting initial drug criticalities...");

  // Example: Set criticalities based on your scenario config (adjust as needed)
  // Assuming Drug 0: Critical (4), Drug 1: High (3), Drug 2: Medium (2)
  const drugCriticalities = [
    { id: 0, value: 4 },
    { id: 1, value: 3 },
    { id: 2, value: 2 },
    // Add more if needed based on --drugs argument in python
  ];

  for (const drug of drugCriticalities) {
    try {
      console.log(`  Setting Drug ${drug.id} criticality to ${drug.value}...`);
      const tx = await supplyChainData.setDrugCriticality(drug.id, drug.value);
      await tx.wait(); // Wait for transaction confirmation
      console.log(`  ✓ Drug ${drug.id} criticality set.`);
    } catch (error) {
       console.error(`  ❌ Failed to set criticality for Drug ${drug.id}:`, error.message);
    }
  }
  console.log("Initial drug criticalities set.");

  // --- Save Artifacts ---
  // Save ABI
  const artifactsPath = path.join(__dirname, "..", "artifacts", "contracts", "SupplyChainData.sol", "SupplyChainData.json");
  const abiPath = path.join(__dirname, "..", "src", "blockchain", "SupplyChainData.abi.json"); // Save ABI where python can find it

  if (fs.existsSync(artifactsPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactsPath, 'utf8'));
    fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));
    console.log(`Contract ABI saved to ${abiPath}`);
  } else {
     console.error(`Error: Artifact file not found at ${artifactsPath}. Compile contracts first.`);
  }

  // Save Address (optional, can also be put in .env)
  const addressPath = path.join(__dirname, "..", "src", "blockchain", "SupplyChainData.address.json");
   fs.writeFileSync(addressPath, JSON.stringify({ address: contractAddress }, null, 2));
   console.log(`Contract address saved to ${addressPath}`);


  // --- Update .env file ---
  updateEnvFile(contractAddress);

}

function updateEnvFile(contractAddress) {
    const envPath = path.join(__dirname, "..", ".env");
    let envContent = "";
    if (fs.existsSync(envPath)) {
        envContent = fs.readFileSync(envPath, "utf8");
    }

    const lines = envContent.split('\n');
    let found = false;
    const newLines = lines.map(line => {
        if (line.startsWith("CONTRACT_ADDRESS=")) {
            found = true;
            return `CONTRACT_ADDRESS="${contractAddress}"`;
        }
        return line;
    }).filter(line => line.trim() !== ''); // Remove empty lines


    if (!found) {
        newLines.push(`CONTRACT_ADDRESS="${contractAddress}"`);
    }

    fs.writeFileSync(envPath, newLines.join('\n') + '\n');
    console.log(`.env file updated with CONTRACT_ADDRESS=${contractAddress}`);
}


main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });