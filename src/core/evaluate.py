import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import pandas as pd

from src.experiments.config import MODEL_CONFIG
from src.utils.utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the evaluator.
        
        Args:
            model (torch.nn.Module): The model to evaluate
            device (str): Device to use for evaluation
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predictions, true labels, and prediction probabilities
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
        
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        preds, labels, probs = self.predict(dataloader)
        
        # Basic classification metrics
        class_report = classification_report(labels, preds, output_dict=True)
        conf_matrix = confusion_matrix(labels, preds)
        
        # ROC and AUC (for binary classification)
        roc_data = None
        if MODEL_CONFIG["num_classes"] == 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            }
            
        # Precision-Recall curve
        pr_curves = {}
        for i in range(MODEL_CONFIG["num_classes"]):
            precision, recall, _ = precision_recall_curve(
                labels == i,
                probs[:, i]
            )
            pr_auc = auc(recall, precision)
            pr_curves[f"class_{i}"] = {
                "precision": precision,
                "recall": recall,
                "auc": pr_auc
            }
            
        return {
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "roc_data": roc_data,
            "pr_curves": pr_curves
        }
        
    def evaluate_and_save(self, dataloader: torch.utils.data.DataLoader,
                         save_dir: Path) -> Dict:
        """
        Evaluate model and save results.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            save_dir (Path): Directory to save evaluation results
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get evaluation metrics
        metrics = self.evaluate(dataloader)
        
        # Save classification report
        pd.DataFrame(metrics["classification_report"]).to_csv(
            save_dir / "classification_report.csv"
        )
        
        # Save confusion matrix
        np.save(save_dir / "confusion_matrix.npy", metrics["confusion_matrix"])
        
        # Save ROC data if available
        if metrics["roc_data"]:
            np.savez(
                save_dir / "roc_data.npz",
                fpr=metrics["roc_data"]["fpr"],
                tpr=metrics["roc_data"]["tpr"],
                auc=metrics["roc_data"]["auc"]
            )
            
        # Save PR curves
        for class_name, pr_data in metrics["pr_curves"].items():
            np.savez(
                save_dir / f"pr_curve_{class_name}.npz",
                precision=pr_data["precision"],
                recall=pr_data["recall"],
                auc=pr_data["auc"]
            )
            
        logger.info(f"Evaluation results saved to {save_dir}")
        return metrics
        
    def analyze_errors(self, dataloader: torch.utils.data.DataLoader) -> pd.DataFrame:
        """
        Analyze prediction errors.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            
        Returns:
            pd.DataFrame: DataFrame containing error analysis
        """
        preds, labels, probs = self.predict(dataloader)
        
        # Get original texts from dataloader
        texts = []
        for batch in dataloader:
            if "text" in batch:
                texts.extend(batch["text"])
            elif "cleaned_text" in batch:
                texts.extend(batch["cleaned_text"])
                
        # Create error analysis DataFrame
        error_df = pd.DataFrame({
            "text": texts,
            "true_label": labels,
            "predicted_label": preds,
            "confidence": np.max(probs, axis=1)
        })
        
        # Add error flag
        error_df["is_error"] = error_df["true_label"] != error_df["predicted_label"]
        
        # Sort by confidence (ascending) to see most uncertain predictions
        error_df = error_df.sort_values("confidence")
        
        return error_df 