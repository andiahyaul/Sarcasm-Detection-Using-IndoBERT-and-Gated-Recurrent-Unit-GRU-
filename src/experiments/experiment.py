import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt

from config import (
    EXPERIMENT_SCENARIOS,
    INDOBERT_CONFIG,
    GRU_CONFIG,
    GRID_SEARCH_PARAMS,
    MODELS_DIR,
    RESULTS_DIR,
    GRID_SEARCH_RESULTS_DIR,
    get_experiment_name,
    get_experiment_path,
    LOG_CONFIG
)
from src.core.dataset import setup_data_pipeline
from src.core.model import SarcasmModel
from src.core.train import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self):
        """Initialize experiment runner"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.current_experiment = None
        self.failed_experiments = []
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Experiment runner initialized on device: {self.device}")
        logger.info(f"Run timestamp: {self.timestamp}")

    def setup_experiment(self, stemming_mode: str, split_mode: str, dimension: int) -> Dict:
        """
        Setup single experiment configuration
        
        Args:
            stemming_mode: "with_stemming" or "without_stemming"
            split_mode: "70_30" or "80_20"
            dimension: Hidden dimension size (512, 768, or 1024)
            
        Returns:
            Dictionary containing experiment configuration
        """
        # Validate inputs
        if stemming_mode not in ["with_stemming", "without_stemming"]:
            raise ValueError(f"Invalid stemming mode: {stemming_mode}")
        if split_mode not in ["70_30", "80_20"]:
            raise ValueError(f"Invalid split mode: {split_mode}")
        if dimension not in [512, 768, 1024]:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        # Create experiment name
        train_ratio = float(split_mode.split("_")[0])/100
        test_ratio = float(split_mode.split("_")[1])/100
        split_ratio = {"train": train_ratio, "test": test_ratio}
        
        preprocess_config = {"name": stemming_mode}
        model_dim = {"name": f"dim_{dimension}"}
        
        experiment_name = get_experiment_name(split_ratio, preprocess_config, model_dim)
        self.current_experiment = experiment_name
        
        # Get experiment paths
        exp_paths = get_experiment_path(experiment_name)
        
        # Create experiment directories
        for path in exp_paths.values():
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for this experiment
        exp_log_path = exp_paths["logs"]
        file_handler = logging.FileHandler(exp_log_path)
        file_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
        logger.addHandler(file_handler)
        
        logger.info(f"Setting up experiment: {experiment_name}")
        logger.info(f"Configuration:")
        logger.info(f"- Stemming mode: {stemming_mode}")
        logger.info(f"- Split mode: {split_mode}")
        logger.info(f"- Dimension: {dimension}")
        
        return {
            "name": experiment_name,
            "paths": exp_paths,
            "config": {
                "stemming_mode": stemming_mode,
                "split_mode": split_mode,
                "dimension": dimension,
                "timestamp": self.timestamp
            }
        }

    def run_grid_search(self, stemming_mode: str, split_mode: str, dimension: int) -> Dict:
        """Run grid search for current experiment configuration"""
        logger.info(f"Starting grid search for {stemming_mode}, {split_mode}, dim={dimension}")
        
        # Create data loaders with a smaller subset for grid search
        train_loader, val_loader, _ = setup_data_pipeline(
            stemming_mode=stemming_mode,
            split_mode=split_mode,
            max_length=INDOBERT_CONFIG["max_length"]
        )
        
        best_val_f1 = 0.0
        best_params = None
        best_history = None
        results = []
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(GRID_SEARCH_PARAMS.keys(), v)) 
                            for v in product(*GRID_SEARCH_PARAMS.values())]
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for params in tqdm(param_combinations, desc="Grid Search"):
            try:
                # Update model config
                current_gru_config = {
                    **GRU_CONFIG,
                    "hidden_sizes": [params["gru_hidden_size"]],
                    "dropout": params["gru_dropout"]
                }
                
                # Initialize model
                model = SarcasmModel(
                    model_config={**INDOBERT_CONFIG, "hidden_size": dimension},
                    gru_config=current_gru_config
                )
                model = model.to(self.device)
                
                # Initialize trainer
                trainer_config = {
                    "learning_rate": params["learning_rate"],
                    "batch_size": params["batch_size"],
                    "weight_decay": params["weight_decay"],
                    "warmup_ratio": params["warmup_ratio"],
                    "patience": params["patience"],
                    "min_delta": params["min_delta"]
                }
                
                trainer = Trainer(model, trainer_config)
                
                # Train and evaluate
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=5
                )
                
                # Get final F1 score
                final_f1 = history["val_metrics"][-1]["weighted avg"]["f1-score"]
                
                results.append({
                    "params": params,
                    "f1_score": final_f1,
                    "history": history
                })
                
                if final_f1 > best_val_f1:
                    best_val_f1 = final_f1
                    best_params = params
                    best_history = history
                    
            except Exception as e:
                logger.error(f"Error in parameter combination {params}: {str(e)}")
                continue
        
        # Save results
        grid_search_results = {
            "best_params": best_params,
            "best_f1": best_val_f1,
            "all_results": results
        }
        
        results_path = GRID_SEARCH_RESULTS_DIR / f"grid_search_{stemming_mode}_{split_mode}_dim{dimension}.json"
        with open(results_path, 'w') as f:
            json.dump(grid_search_results, f, indent=4)
            
        # Automatically create plot with best parameters
        self._create_progress_plot(
            scenario_num=2 if split_mode == "80_20" else 1,
            split_ratio=split_mode,
            history=best_history,
            best_params=best_params
        )
        
        return grid_search_results
        
    def _create_progress_plot(self, scenario_num: int, split_ratio: str, history: Dict, best_params: Dict):
        """Create progress plot with best parameters"""
        plt.figure(figsize=(15, 6))

        # 1. Training Progress
        plt.subplot(1, 2, 1)
        epochs = range(1, len(history['train_loss']) + 1)

        # Get F1-scores for each epoch
        f1_scores = [metrics['weighted avg']['f1-score'] for metrics in history['val_metrics']]

        # Plot F1-scores with value labels
        plt.plot(epochs, f1_scores, 'r-o', label='F1-Score', linewidth=2, markersize=8)
        for i, score in enumerate(f1_scores, 1):
            plt.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')

        plt.title(f'Scenario {scenario_num} - F1-Score Progress ({split_ratio} split)', fontsize=12, pad=15)
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Adjust y-axis to make room for labels
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax + 0.05)

        # 2. Best Parameters and Final Metrics
        plt.subplot(1, 2, 2)
        plt.axis('off')

        # Get final metrics
        final_metrics = history['val_metrics'][-1]
        
        # Display only the four main parameters
        display_params = {
            "Hidden Size": best_params["gru_hidden_size"],
            "Dropout": best_params["gru_dropout"],
            "Learning Rate": best_params["learning_rate"],
            "Batch Size": best_params["batch_size"]
        }

        # Create text for display
        param_text = f"Scenario {scenario_num} Best Parameters:\n\n"
        for param, value in display_params.items():
            param_text += f"{param}: {value}\n"

        param_text += f"\nF1-Scores by Epoch:\n"
        for epoch, score in enumerate(f1_scores, 1):
            param_text += f"Epoch {epoch}: {score:.4f}\n"

        param_text += f"\nFinal Metrics:\n"
        param_text += f"Accuracy: {final_metrics['accuracy']:.4f}\n"
        param_text += f"F1-Score: {final_metrics['weighted avg']['f1-score']:.4f}\n"
        param_text += f"Precision: {final_metrics['weighted avg']['precision']:.4f}\n"
        param_text += f"Recall: {final_metrics['weighted avg']['recall']:.4f}"

        plt.text(0.5, 0.5, param_text, 
                 fontsize=12, 
                 va='center', 
                 ha='center',
                 bbox=dict(facecolor='white', 
                          alpha=0.8, 
                          edgecolor='gray', 
                          boxstyle='round,pad=1'))

        plt.tight_layout()
        plt.savefig(f'plots/scenario_{scenario_num}_progress.png', dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved as 'plots/scenario_{scenario_num}_progress.png'")

    def run_single_experiment(self, stemming_mode: str, split_mode: str, dimension: int) -> Dict:
        """Run single experiment without grid search"""
        start_time = datetime.now()
        
        try:
            # Setup experiment
            exp_config = self.setup_experiment(stemming_mode, split_mode, dimension)
            logger.info(f"Starting experiment: {exp_config['name']}")
            
            # Use best parameters directly
            best_params = {
                "gru_hidden_size": 768,
                "gru_dropout": 0.1,
                "learning_rate": 2e-5,
                "gru_bidirectional": True,
                "gru_num_layers": 2,
                "batch_size": 32,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "patience": 2,
                "min_delta": 1e-4
            }
            
            # Create data loaders
            train_loader, val_loader, test_loader = setup_data_pipeline(
                stemming_mode=stemming_mode,
                split_mode=split_mode,
                max_length=INDOBERT_CONFIG["max_length"]
            )
            
            # Initialize model with best params
            model_config = {**INDOBERT_CONFIG, "hidden_size": dimension}
            gru_config = {
                **GRU_CONFIG,
                "hidden_sizes": [best_params["gru_hidden_size"]],
                "num_layers": best_params["gru_num_layers"],
                "dropout": best_params["gru_dropout"],
                "bidirectional": best_params["gru_bidirectional"]
            }
            
            model = SarcasmModel(model_config, gru_config)
            model = model.to(self.device)
            
            # Train with best params
            trainer_config = {
                "learning_rate": best_params["learning_rate"],
                "batch_size": best_params["batch_size"],
                "weight_decay": best_params["weight_decay"],
                "warmup_ratio": best_params["warmup_ratio"],
                "patience": best_params["patience"],
                "min_delta": best_params["min_delta"]
            }
            
            trainer = Trainer(model, trainer_config)
            
            # Full training
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=5,  # Changed to 5 epochs
                save_dir=exp_config["paths"]["model"]
            )
            
            # Evaluate
            test_loss, test_metrics = trainer.evaluate(test_loader)
            
            # Compile results
            results = {
                "experiment_name": exp_config["name"],
                "config": exp_config["config"],
                "best_params": best_params,
                "history": history,
                "test_metrics": test_metrics,
                "test_loss": test_loss,
                "duration": (datetime.now() - start_time).total_seconds()
            }
            
            self.save_experiment_results(results, exp_config["paths"])
            return results
            
        except Exception as e:
            logger.error(f"Error in experiment: {str(e)}")
            self.failed_experiments.append({
                "config": {"stemming_mode": stemming_mode, "split_mode": split_mode, "dimension": dimension},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise

    def run_all_experiments(self):
        """Run all experiment scenarios"""
        logger.info("Starting all experiments")
        start_time = datetime.now()
        
        # Generate all experiment combinations
        experiments = [
            ("with_stemming", "70_30", 768),    # Scenario 1
            ("with_stemming", "80_20", 768),    # Scenario 2
            ("without_stemming", "70_30", 768), # Scenario 3
            ("without_stemming", "80_20", 768)  # Scenario 4
        ]
        
        progress_bar = tqdm(experiments, desc="Running experiments")
        for config in progress_bar:
            try:
                stemming_mode, split_mode, dimension = config
                progress_bar.set_description(
                    f"Running experiment: {stemming_mode}, {split_mode}, dim={dimension}"
                )
                
                results = self.run_single_experiment(
                    stemming_mode=stemming_mode,
                    split_mode=split_mode,
                    dimension=dimension
                )
                self.results.append(results)
                
            except Exception as e:
                logger.error(f"Experiment failed: {config}")
                logger.error(f"Error: {str(e)}")
                continue
        
        # Calculate total duration
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Save all results
        self.save_all_results(total_duration)
        
        # Log summary
        logger.info("\nExperiment Summary:")
        logger.info(f"Total experiments run: {len(experiments)}")
        logger.info(f"Successful experiments: {len(self.results)}")
        logger.info(f"Failed experiments: {len(self.failed_experiments)}")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        
        if self.failed_experiments:
            logger.warning("\nFailed Experiments:")
            for failed in self.failed_experiments:
                logger.warning(f"Config: {failed['config']}")
                logger.warning(f"Error: {failed['error']}")

    def save_experiment_results(self, results: Dict, paths: Dict):
        """
        Save individual experiment results
        
        Args:
            results: Experiment results
            paths: Experiment paths
        """
        # Save metrics
        metrics_path = paths["metrics"]
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save training history plot
        if "history" in results:
            plot_path = paths["plots"] / "training_history.png"
            self.plot_training_history(results["history"], plot_path)
        
        logger.info(f"Results saved to {metrics_path}")

    def save_all_results(self, total_duration: float):
        """
        Save results from all experiments
        
        Args:
            total_duration: Total duration of all experiments
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create summary DataFrame
        summary = []
        for result in self.results:
            summary.append({
                "experiment_name": result["experiment_name"],
                "stemming_mode": result["config"]["stemming_mode"],
                "split_mode": result["config"]["split_mode"],
                "dimension": result["config"]["dimension"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_f1": result["test_metrics"]["weighted avg"]["f1-score"],
                "duration": result["duration"]
            })
        
        summary_df = pd.DataFrame(summary)
        
        # Save summary
        summary_path = RESULTS_DIR / f"experiment_summary_{self.timestamp}.json"
        summary_df.to_json(summary_path, orient="records", indent=4)
        
        # Save failed experiments
        if self.failed_experiments:
            failed_path = RESULTS_DIR / f"failed_experiments_{self.timestamp}.json"
            with open(failed_path, 'w') as f:
                json.dump(self.failed_experiments, f, indent=4)
        
        # Create and save final report
        report = {
            "timestamp": self.timestamp,
            "total_experiments": len(self.results) + len(self.failed_experiments),
            "successful_experiments": len(self.results),
            "failed_experiments": len(self.failed_experiments),
            "total_duration": total_duration,
            "best_experiment": self.get_best_experiment()["experiment_name"],
            "device": str(self.device)
        }
        
        report_path = RESULTS_DIR / f"experiment_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"All results saved to {RESULTS_DIR}")

    def get_best_experiment(self) -> Dict:
        """
        Get best performing experiment
        
        Returns:
            Dictionary containing best experiment results
        """
        if not self.results:
            raise ValueError("No experiments have been run")
        
        best_result = max(
            self.results,
            key=lambda x: x["test_metrics"]["accuracy"]
        )
        
        return best_result

    def plot_training_history(self, history: Dict, save_path: Path):
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot([m["accuracy"] for m in history["val_metrics"]])
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 