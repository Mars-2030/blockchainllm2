

this is my code structure:
pandemic_supply_chain/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── src/
│   ├── init.py
│   ├── scenario/
│   │   ├── init.py
│   │   ├── generator.py
│   │   └── visualizer.py
│   ├── environment/
│   │   ├── init.py
│   │   ├── supply_chain.py
│   │   └── metrics.py
│   ├── agents/
│   │   ├── init.py
│   │   ├── base.py
│   │   ├── manufacturer.py
│   │   ├── distributor.py
│   │   └── hospital.py
│   ├── tools/
│   │   ├── init.py
│   │   ├── forecasting.py
│   │   ├── allocation.py
│   │   └── assessment.py
│   ├── llm/
│   │   ├── init.py
│   │   └── openai_integration.py
│   └── blockchain/
│       ├── init.py
│       └── interface.py
└── output/
└── .gitkeep