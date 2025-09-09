import os
from pathlib import Path
from itertools import product

# Project paths
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Visualization directories
PLOTS_DIR = RESULTS_DIR / "plots"
TRAIN_PLOTS_DIR = PLOTS_DIR / "training"
EVAL_PLOTS_DIR = PLOTS_DIR / "evaluation"
GRID_SEARCH_PLOTS_DIR = PLOTS_DIR / "grid_search"
COMPARISON_PLOTS_DIR = PLOTS_DIR / "comparison"

# Specific plot directories
LOSS_PLOTS_DIR = TRAIN_PLOTS_DIR / "loss_curves"
ACCURACY_PLOTS_DIR = TRAIN_PLOTS_DIR / "accuracy_curves"
CONFUSION_MATRIX_DIR = EVAL_PLOTS_DIR / "confusion_matrices"
ROC_CURVES_DIR = EVAL_PLOTS_DIR / "roc_curves"
PR_CURVES_DIR = EVAL_PLOTS_DIR / "pr_curves"
ERROR_ANALYSIS_DIR = EVAL_PLOTS_DIR / "error_analysis"
PARAMETER_STUDY_DIR = GRID_SEARCH_PLOTS_DIR / "parameter_study"

# Report directories
REPORTS_DIR = RESULTS_DIR / "reports"
TRAINING_LOGS_DIR = REPORTS_DIR / "training_logs"
EVAL_METRICS_DIR = REPORTS_DIR / "evaluation_metrics"
GRID_SEARCH_RESULTS_DIR = REPORTS_DIR / "grid_search_results"
EXPERIMENT_RESULTS_DIR = REPORTS_DIR / "experiment_results"

# Experiment Scenarios Configuration
EXPERIMENT_SCENARIOS = {
    "split_ratios": [
        {"train": 0.7, "val": 0.15, "test": 0.15},  # 70:30 split
        {"train": 0.8, "val": 0.1, "test": 0.1}     # 80:20 split
    ],
    "preprocessing_configs": [
        {"use_stemming": True, "name": "with_stemming"},
        {"use_stemming": False, "name": "no_stemming"}
    ],
    "model_dimensions": [
        {"hidden_size": 768, "name": "dim_768"}  # Match IndoBERT's dimension
    ]
}

# Generate all experiment combinations
EXPERIMENT_COMBINATIONS = list(product(
    EXPERIMENT_SCENARIOS["split_ratios"],
    EXPERIMENT_SCENARIOS["preprocessing_configs"],
    EXPERIMENT_SCENARIOS["model_dimensions"]
))

# Export paths configuration
EXPORT_PATHS = {
    # Model checkpoints and saved models
    "model_checkpoint": MODELS_DIR / "checkpoint_{epoch:03d}.pt",
    "best_model": MODELS_DIR / "best_model.pt",
    "final_model": MODELS_DIR / "final_model.pt",
    
    # Predictions and results
    "predictions": RESULTS_DIR / "predictions.csv",
    "error_analysis": RESULTS_DIR / "error_analysis.csv",
    
    # Evaluation metrics
    "metrics_summary": EVAL_METRICS_DIR / "summary.json",
    "confusion_matrix": EVAL_METRICS_DIR / "confusion_matrix.npy",
    "classification_report": EVAL_METRICS_DIR / "classification_report.csv",
    
    # Training history
    "training_history": TRAINING_LOGS_DIR / "training_history.json",
    "learning_curves": TRAIN_PLOTS_DIR / "learning_curves.png",
    
    # Grid search results
    "grid_search_results": GRID_SEARCH_RESULTS_DIR / "grid_search_results.json",
    "best_params": GRID_SEARCH_RESULTS_DIR / "best_params.json",
    
    # Experiment results
    "experiment_summary": EXPERIMENT_RESULTS_DIR / "experiment_summary.json",
    "experiment_comparison": EXPERIMENT_RESULTS_DIR / "experiment_comparison.csv",
    "experiment_plots": COMPARISON_PLOTS_DIR / "scenario_comparison.png"
}

# Data file paths
CLEAN_SARCASM_PATH = DATA_DIR / "raw" / "clean_sarcasm_data.csv"
SLANG_WORDS_PATH = DATA_DIR / "raw" / "slang_words (2).csv"

# Create all directories
DIRS_TO_CREATE = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_PLOTS_DIR,
    EVAL_PLOTS_DIR,
    GRID_SEARCH_PLOTS_DIR,
    COMPARISON_PLOTS_DIR,
    LOSS_PLOTS_DIR,
    ACCURACY_PLOTS_DIR,
    CONFUSION_MATRIX_DIR,
    ROC_CURVES_DIR,
    PR_CURVES_DIR,
    ERROR_ANALYSIS_DIR,
    PARAMETER_STUDY_DIR,
    TRAINING_LOGS_DIR,
    EVAL_METRICS_DIR,
    GRID_SEARCH_RESULTS_DIR,
    EXPERIMENT_RESULTS_DIR
]

for dir_path in DIRS_TO_CREATE:
    dir_path.mkdir(parents=True, exist_ok=True)

# Verify data files exist
for data_path in [CLEAN_SARCASM_PATH, PREPROCESSED_SARCASM_PATH, SLANG_WORDS_PATH]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

# IndoBERT Configuration
INDOBERT_CONFIG = {
    "model_name": "indobenchmark/indobert-base-p1",
    "hidden_size": 768,
    "max_length": 128,     # Further reduced from 256
    "dropout": 0.1,
    "num_classes": 2,
    "freeze_bert": True,
    "batch_size": 32      # Increased for better parallelization
}

# GRU Configuration
GRU_CONFIG = {
    "hidden_sizes": [384],  # Reduced from 768
    "num_layers": 1,
    "dropout": 0.1,
    "bidirectional": False,
    "batch_first": True
}

# Tokenizer Configuration
TOKENIZER_CONFIG = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt"
}

# Dataset Configuration
DATASET_CONFIG = {
    "split_70_30": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    "split_80_20": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
}

# DataLoader Configuration
DATALOADER_CONFIG = {
    "batch_size": 32,      # Increased for GPU
    "shuffle": True,
    "num_workers": 2,      # Re-enable workers for GPU
    "pin_memory": True,    # Enable pin_memory for GPU
    "persistent_workers": True  # Keep workers alive between epochs
}

# Training Configuration
TRAINING_CONFIG = {
    "loss": {
        "name": "CrossEntropyLoss",
        "params": {}
    },
    "optimizer": {
        "name": "AdamW",
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "name": "OneCycleLR",
        "params": {
            "max_lr": 2e-5,
            "pct_start": 0.1,
            "anneal_strategy": "linear"
        }
    }
}

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    "patience": 2,
    "min_delta": 1e-4
}

# Checkpoint Configuration
CHECKPOINT_CONFIG = {
    "save_best": True,
    "save_last": True,
    "save_interval": 5
}

# Base Training Configuration
BASE_TRAIN_CONFIG = {
    "num_epochs": 5,  # Changed to 5 epochs
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "eval_steps": 100,
    "save_steps": 1000,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "use_gpu": True,
    "mixed_precision": True
}

# Grid Search Parameters
GRID_SEARCH_PARAMS = {
    "gru_hidden_size": [768],          # Fixed hidden size
    "learning_rate": [2e-5],           # Different from scenario 3
    "gru_dropout": [0.2],              # Different from all previous scenarios
    "batch_size": [16],                # Different from all previous scenarios
    # Keep other parameters fixed
    "gru_bidirectional": [False],
    "gru_num_layers": [1],
    "weight_decay": [0.01],
    "warmup_ratio": [0.1],
    "patience": [2],
    "min_delta": [1e-4]
}

# Base Preprocessing Configuration
BASE_PREPROCESS_CONFIG = {
    "remove_punctuation": True,
    "remove_numbers": False,
    "remove_urls": True,
    "remove_emails": True,
    "lowercase": True,
    "remove_slang": True  # Enable slang word normalization
}

def get_experiment_name(split_ratio, preprocess_config, model_dim):
    """Generate a unique name for each experiment scenario."""
    split_name = f"split_{int(split_ratio['train']*100)}-{int(split_ratio['test']*100)}"
    stem_name = preprocess_config['name']
    dim_name = model_dim['name']
    return f"{split_name}_{stem_name}_{dim_name}"

def get_experiment_path(experiment_name):
    """Generate paths for experiment results."""
    exp_dir = EXPERIMENT_RESULTS_DIR / experiment_name
    return {
        "model": MODELS_DIR / experiment_name,
        "logs": TRAINING_LOGS_DIR / f"{experiment_name}.log",
        "metrics": EVAL_METRICS_DIR / f"{experiment_name}.json",
        "plots": PLOTS_DIR / experiment_name,
        "results": exp_dir
    }

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": TRAINING_LOGS_DIR / "training.log"
}

# Visualization Configuration
VIZ_CONFIG = {
    "style": "seaborn",
    "figsize": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "plot_dirs": {
        "training": {
            "loss": LOSS_PLOTS_DIR,
            "accuracy": ACCURACY_PLOTS_DIR
        },
        "evaluation": {
            "confusion_matrix": CONFUSION_MATRIX_DIR,
            "roc_curves": ROC_CURVES_DIR,
            "pr_curves": PR_CURVES_DIR,
            "error_analysis": ERROR_ANALYSIS_DIR
        },
        "grid_search": {
            "parameter_study": PARAMETER_STUDY_DIR
        },
        "comparison": COMPARISON_PLOTS_DIR
    }
} 