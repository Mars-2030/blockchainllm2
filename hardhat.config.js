require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config(); // To load .env file

// const SEPOLIA_RPC_URL = process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org";
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0xabcdef..."; // Provide a default or ensure it's in .env
// const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY || "";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  defaultNetwork: "hardhat", // Use hardhat local network by default
  networks: {
    hardhat: {
      chainId: 31337, // Standard chain ID for Hardhat Network
      // forking: { // Optional: Fork mainnet or testnet
      //   url: MAINNET_RPC_URL
      // }
    },
    localhost: {
      chainId: 31337, // Or 1337 if using default Ganache
      url: "http://127.0.0.1:8545/", // Default RPC for Hardhat node and Ganache
      // accounts: [PRIVATE_KEY], // Hardhat node usually provides accounts
    },
    // sepolia: {
    //   url: SEPOLIA_RPC_URL,
    //   accounts: [`0x${PRIVATE_KEY}`], // Ensure '0x' prefix
    //   chainId: 11155111,
    // },
  },
  //  etherscan: {
  //    apiKey: {
  //        sepolia: ETHERSCAN_API_KEY,
  //    }
  //  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts", // Output directory for ABI and bytecode
  },
  mocha: {
    timeout: 40000 // Increase timeout for tests if needed
  }
};