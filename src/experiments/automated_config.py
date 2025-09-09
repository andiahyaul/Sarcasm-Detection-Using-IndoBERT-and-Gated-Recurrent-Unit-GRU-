import torch
from pathlib import Path
from itertools import product

# Device Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Model Configuration
MODEL_CONFIG = {
    "model_name": "indobenchmark/indobert-base-p1",
    "hidden_size": 768,
    "num_classes": 2
}

# Experiment Structure
SCENARIOS = [
    {"split": "70_30", "stemming": True},   # Scenario 1: 70/30 split with stemming
    {"split": "70_30", "stemming": False},  # Scenario 2: 70/30 split without stemming
    {"split": "80_20", "stemming": True},   # Scenario 3: 80/20 split with stemming
    {"split": "80_20", "stemming": False}   # Scenario 4: 80/20 split without stemming
]

# Training Configuration
TRAINING_CONFIG = {
    "num_epochs": 5,  # Fixed at 5 epochs
    "early_stopping_patience": 10,  # Set high to effectively disable early stopping
    "gradient_clipping": 1.0,
    "weight_decay": 0.01,
    "min_epochs": 5  # Force training to run all 5 epochs
}

# Grid Search Parameter Space
PARAMETER_GRID = {
    "hidden_size": [128, 256, 512],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [0.0001, 0.0003, 0.001],
    "batch_size": [16, 32],
    "gru_layers": [1, 2]
}

# Number of combinations per scenario
COMBINATIONS_PER_SCENARIO = 5

# Directory Structure
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
EXPERIMENT_DIR = BASE_DIR / "grid_search_experiments"
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
LOGS_DIR = EXPERIMENT_DIR / "logs"

# Create directories
for dir_path in [EXPERIMENT_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def generate_grid_combinations():
    """Generate systematic grid combinations"""
    all_combinations = []
    for values in product(
        PARAMETER_GRID["hidden_size"],
        PARAMETER_GRID["dropout"],
        PARAMETER_GRID["learning_rate"],
        PARAMETER_GRID["batch_size"],
        PARAMETER_GRID["gru_layers"]
    ):
        params = {
            "hidden_size": values[0],
            "dropout": values[1],
            "learning_rate": values[2],
            "batch_size": values[3],
            "gru_layers": values[4]
        }
        all_combinations.append(params)
    
    # Select first COMBINATIONS_PER_SCENARIO combinations
    return all_combinations[:COMBINATIONS_PER_SCENARIO]

def get_experiment_name(scenario_idx, iteration, params):
    """Generate unique experiment name"""
    scenario = SCENARIOS[scenario_idx]
    split = scenario["split"]
    stem = "stem" if scenario["stemming"] else "nostem"
    return f"grid_exp_{split}_{stem}_iter{iteration}_h{params['hidden_size']}_d{params['dropout']}_lr{params['learning_rate']}_b{params['batch_size']}_l{params['gru_layers']}" 