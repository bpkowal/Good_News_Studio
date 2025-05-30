RAGAIMODEL/
├── agent_outputs/
│   ├── response_20250413_182447.txt
│   └── ...
│
├── care_ethics_corpus/
│   ├── relational_ethics.md
│   ├── emotional_protection.md
│   └── ...
│
├── utilitarian_corpus/
│   ├── aggregate_welfare.md
│   ├── sacrifice.md
│   └── ...
│
├── scenarios/
│   ├── auto_ethics_1727_20250413.json
│   ├── auto_care_ethics_0837_20250413.json
│   └── ...
│
├── care_ethics_agent.py
├── care_ethics_critic.py
├── utilitarian_agent.py
├── utilitarian_critic.py
├── general_scenario_builder.py
├── care_scenario_builder.py
├── get_semantic_tag_weights.py
├── load_care_ethics_corpus.py
├── load_utilitarian_corpus.py
└── requirements.txt

This is the model so far. get_semantic_tag.py is a work in progress and may need to be fixed into the care care_ethics_agent
It turns out the tags for the scenarios matter a lot for finding the correct quotes for each agent . You probably want to match each a seperate 
set of tags for each agent.