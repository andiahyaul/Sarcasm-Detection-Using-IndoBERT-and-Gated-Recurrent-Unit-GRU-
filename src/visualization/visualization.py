import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix

from src.experiments.config import VIZ_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationManager:
    def __init__(self, save_dir: Path):
        """
        Initialize visualization manager.
        
        Args:
            save_dir (Path): Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(VIZ_CONFIG["style"])
        
    def plot_training_history(self, history: Dict[str, List[float]], save_name: str = "training_history"):
        """
        Plot training metrics history.
        
        Args:
            history (Dict[str, List[float]]): Training history dictionary
            save_name (str): Name for saving the plot
        """
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        
        # Plot training loss
        plt.plot(history["train_loss"], label="Training Loss", marker="o")
        plt.plot(history["val_loss"], label="Validation Loss", marker="o")
        
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(
            self.save_dir / f"{save_name}_loss.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close()
        
        # Plot accuracy if available
        if "val_accuracy" in history:
            plt.figure(figsize=VIZ_CONFIG["figsize"])
            plt.plot(history["val_accuracy"], label="Validation Accuracy", marker="o")
            
            plt.title("Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            
            plt.savefig(
                self.save_dir / f"{save_name}_accuracy.{VIZ_CONFIG['save_format']}",
                dpi=VIZ_CONFIG["dpi"],
                bbox_inches="tight"
            )
            plt.close()
            
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_name: str = "confusion_matrix"):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix
            class_names (List[str], optional): List of class names
            save_name (str): Name for saving the plot
        """
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        
        # Create heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto"
        )
        
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        plt.savefig(
            self.save_dir / f"{save_name}.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close()
        
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float,
                      save_name: str = "roc_curve"):
        """
        Plot ROC curve.
        
        Args:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates
            auc (float): Area under curve
            save_name (str): Name for saving the plot
        """
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(
            self.save_dir / f"{save_name}.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close()
        
    def plot_pr_curves(self, pr_curves: Dict[str, Dict],
                      save_name: str = "pr_curves"):
        """
        Plot precision-recall curves.
        
        Args:
            pr_curves (Dict[str, Dict]): Dictionary containing PR curves for each class
            save_name (str): Name for saving the plot
        """
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        
        for class_name, pr_data in pr_curves.items():
            plt.plot(
                pr_data["recall"],
                pr_data["precision"],
                label=f"{class_name} (AUC = {pr_data['auc']:.3f})"
            )
            
        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(
            self.save_dir / f"{save_name}.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close()
        
    def plot_grid_search_results(self, results: List[Dict],
                               save_name: str = "grid_search"):
        """
        Plot grid search results.
        
        Args:
            results (List[Dict]): List of grid search results
            save_name (str): Name for saving the plot
        """
        # Convert results to DataFrame
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        # Melt parameters into long format
        param_df = pd.json_normalize(results_df["params"])
        results_df = pd.concat([
            results_df.drop("params", axis=1),
            param_df
        ], axis=1)
        
        # Create interactive parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=results_df["val_accuracy"],
                    colorscale="Viridis"
                ),
                dimensions=[
                    dict(range=[results_df[col].min(), results_df[col].max()],
                         label=col,
                         values=results_df[col])
                    for col in results_df.columns
                ]
            )
        )
        
        fig.update_layout(
            title="Grid Search Results",
            width=1200,
            height=800
        )
        
        # Save interactive plot
        fig.write_html(self.save_dir / f"{save_name}.html")
        
        # Create static plot for parameters vs validation accuracy
        for param in param_df.columns:
            plt.figure(figsize=VIZ_CONFIG["figsize"])
            
            plt.scatter(param_df[param], results_df["val_accuracy"])
            plt.xlabel(param)
            plt.ylabel("Validation Accuracy")
            plt.title(f"Validation Accuracy vs {param}")
            plt.grid(True)
            
            plt.savefig(
                self.save_dir / f"{save_name}_{param}.{VIZ_CONFIG['save_format']}",
                dpi=VIZ_CONFIG["dpi"],
                bbox_inches="tight"
            )
            plt.close()
            
    def plot_error_analysis(self, error_df: pd.DataFrame,
                           save_name: str = "error_analysis"):
        """
        Plot error analysis visualizations.
        
        Args:
            error_df (pd.DataFrame): DataFrame containing error analysis
            save_name (str): Name for saving the plot
        """
        # Confidence distribution for correct vs incorrect predictions
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        
        sns.histplot(
            data=error_df,
            x="confidence",
            hue="is_error",
            multiple="stack",
            bins=30
        )
        
        plt.title("Confidence Distribution for Correct vs Incorrect Predictions")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        
        plt.savefig(
            self.save_dir / f"{save_name}_confidence.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close()
        
        # Error rate by confidence bucket
        error_df["confidence_bucket"] = pd.qcut(error_df["confidence"], q=10)
        error_rate = error_df.groupby("confidence_bucket")["is_error"].mean()
        
        plt.figure(figsize=VIZ_CONFIG["figsize"])
        error_rate.plot(kind="bar")
        
        plt.title("Error Rate by Confidence Level")
        plt.xlabel("Confidence Bucket")
        plt.ylabel("Error Rate")
        plt.xticks(rotation=45)
        
        plt.savefig(
            self.save_dir / f"{save_name}_error_rate.{VIZ_CONFIG['save_format']}",
            dpi=VIZ_CONFIG["dpi"],
            bbox_inches="tight"
        )
        plt.close() 