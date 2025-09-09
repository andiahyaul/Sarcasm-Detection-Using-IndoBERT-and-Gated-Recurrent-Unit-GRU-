import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import json
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import gc

from src.experiments.automated_config import *
from src.core.model import SarcasmModel
from src.core.dataset import setup_data_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'automated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTrainer:
    def __init__(self):
        """Initialize automated trainer"""
        self.device = torch.device(DEVICE)
        self.results = []
        self.current_experiment = None
        self.failed_experiments = []
        self.start_time = datetime.now()
        
        logger.info(f"Initialized AutomatedTrainer on device: {self.device}")
        logger.info(f"GPU Available: {torch.backends.mps.is_available()}")
        logger.info(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    def train_single_experiment(self, scenario_idx, iteration):
        """Train a single experiment with random parameters"""
        scenario = SCENARIOS[scenario_idx]
        params = generate_random_params()
        exp_name = get_experiment_name(scenario_idx, iteration, params)
        
        logger.info(f"Starting experiment: {exp_name}")
        logger.info(f"Parameters: {params}")
        
        # Setup data pipeline
        train_loader, val_loader, test_loader = setup_data_pipeline(
            stemming_mode="with_stemming" if scenario["stemming"] else "without_stemming",
            split_mode=scenario["split"],
            max_length=128,
            batch_size=params["batch_size"]
        )
        
        # Initialize model with GRU config from params
        gru_config = {
            "hidden_sizes": [params["hidden_size"]],
            "num_layers": params["gru_layers"],
            "dropout": params["dropout"],
            "bidirectional": False
        }
        
        model = SarcasmModel(model_config=MODEL_CONFIG, gru_config=gru_config)
        model = model.to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_metrics": []
        }
        
        for epoch in range(TRAINING_CONFIG["num_epochs"]):
            # Training
            model.train()
            train_losses = []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clipping"])
                optimizer.step()
                
                train_losses.append(loss.item())
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Validation
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_losses.append(loss.item())
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            from sklearn.metrics import classification_report
            val_metrics = classification_report(val_labels, val_preds, output_dict=True)
            val_f1 = val_metrics["weighted avg"]["f1-score"]
            
            # Update history
            history["train_loss"].append(np.mean(train_losses))
            history["val_loss"].append(np.mean(val_losses))
            history["val_f1"].append(val_f1)
            history["val_metrics"].append(val_metrics)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {np.mean(train_losses):.4f}, "
                      f"Val Loss: {np.mean(val_losses):.4f}, Val F1: {val_f1:.4f}")
            
            # Save intermediate results
            intermediate_results = {
                "experiment_name": exp_name,
                "scenario": scenario,
                "params": params,
                "history": history,
                "best_f1": best_f1,
                "current_epoch": epoch + 1,
                "final_metrics": val_metrics
            }
            self.save_experiment_results(intermediate_results)
            
            # Early stopping check (but will complete minimum epochs)
            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save checkpoint
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "params": params,
                    "metrics": val_metrics,
                    "epoch": epoch + 1
                }, CHECKPOINTS_DIR / f"{exp_name}_best.pt")
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
            
            # Log progress
            logger.info(f"Completed epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']}")
            logger.info(f"Current F1: {val_f1:.4f}, Best F1: {best_f1:.4f}")
            
            # Force training to continue for all epochs
            if epoch + 1 < TRAINING_CONFIG["num_epochs"]:
                logger.info(f"Continuing training: {TRAINING_CONFIG['num_epochs'] - (epoch + 1)} epochs remaining")
        
        # Save results
        results = {
            "experiment_name": exp_name,
            "scenario": scenario,
            "params": params,
            "history": history,
            "best_f1": best_f1,
            "final_metrics": history["val_metrics"][-1]
        }
        
        self.results.append(results)
        self.save_experiment_results(results)
        
        # Clear memory
        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
        
    def run_all_experiments(self):
        """Run all experiments"""
        total_experiments = len(SCENARIOS) * ITERATIONS_PER_SCENARIO
        logger.info(f"Starting {total_experiments} experiments")
        
        for scenario_idx in range(len(SCENARIOS)):
            for iteration in range(ITERATIONS_PER_SCENARIO):
                logger.info(f"Running scenario {scenario_idx+1}, iteration {iteration+1}")
                self.train_single_experiment(scenario_idx, iteration)
        
        self.create_comparison_plot()
        self.save_final_report()

    def save_experiment_results(self, results):
        """Save individual experiment results"""
        exp_path = RESULTS_DIR / f"{results['experiment_name']}.json"
        with open(exp_path, 'w') as f:
            json.dump(results, f, indent=4)

    def create_comparison_plot(self):
        """Create comparison plot of all experiments"""
        plt.figure(figsize=(15, 10))
        
        # Prepare data for plotting
        scenario_results = {i: [] for i in range(len(SCENARIOS))}
        for result in self.results:
            scenario_idx = SCENARIOS.index(result["scenario"])
            scenario_results[scenario_idx].append(result["best_f1"])
        
        # Box plot
        plt.subplot(2, 1, 1)
        data = [results for results in scenario_results.values()]
        labels = [f"Scenario {i+1}" for i in range(len(SCENARIOS))]
        plt.boxplot(data, labels=labels)
        plt.title("F1-Score Distribution Across Scenarios")
        plt.ylabel("F1-Score")
        
        # Line plot of best runs
        plt.subplot(2, 1, 2)
        for scenario_idx, results in scenario_results.items():
            best_result = max(
                [r for r in self.results if SCENARIOS.index(r["scenario"]) == scenario_idx],
                key=lambda x: x["best_f1"]
            )
            f1_scores = best_result["history"]["val_f1"]
            plt.plot(range(1, len(f1_scores) + 1), f1_scores, 
                    label=f"Scenario {scenario_idx+1}", marker='o')
        
        plt.title("Best F1-Score Progress per Scenario")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scenario_comparison.png", dpi=300, bbox_inches='tight')
        logger.info("Comparison plot saved")

    def save_final_report(self):
        """Save final report with all results"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            "total_experiments": len(SCENARIOS) * ITERATIONS_PER_SCENARIO,
            "successful_experiments": len(self.results),
            "failed_experiments": len(self.failed_experiments),
            "duration_seconds": duration,
            "device": str(self.device),
            "best_results": {},
            "failed_experiments": self.failed_experiments
        }
        
        # Add best results for each scenario
        for scenario_idx in range(len(SCENARIOS)):
            scenario_results = [r for r in self.results 
                              if SCENARIOS.index(r["scenario"]) == scenario_idx]
            if scenario_results:
                best_result = max(scenario_results, key=lambda x: x["best_f1"])
                report["best_results"][f"scenario_{scenario_idx+1}"] = {
                    "experiment_name": best_result["experiment_name"],
                    "params": best_result["params"],
                    "best_f1": best_result["best_f1"],
                    "final_metrics": best_result["final_metrics"]
                }
        
        # Save report
        with open(RESULTS_DIR / "final_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        logger.info("Final report saved")

if __name__ == "__main__":
    trainer = AutomatedTrainer()
    trainer.run_all_experiments() 