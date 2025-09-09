import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import signal
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import psutil
from torch.cuda.amp import GradScaler, autocast

from src.experiments.config import (
    TRAINING_CONFIG,
    EARLY_STOPPING_CONFIG,
    CHECKPOINT_CONFIG,
    MODELS_DIR
)
from src.core.model import SarcasmModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_CONFIG["patience"], 
                 min_delta=EARLY_STOPPING_CONFIG["min_delta"]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        logger.info(f"Initialized early stopping - Patience: {patience}, Min delta: {min_delta}")
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered")
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.stop_training = False
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Setup device (MPS for Apple Silicon, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
            
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = getattr(nn, TRAINING_CONFIG["loss"]["name"])(**TRAINING_CONFIG["loss"]["params"])
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            **TRAINING_CONFIG["optimizer"]["params"]
        )
        
        self.best_val_loss = float('inf')
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal"""
        print("\nUser aborted training. Saving checkpoint and cleaning up...")
        self.stop_training = True

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """
        Training for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader, phase: str = "val") -> Tuple[float, Dict]:
        """
        Evaluate model
        
        Args:
            val_loader: Validation/Test data loader
            phase: "val" or "test"
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Evaluating ({phase})")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        metrics = classification_report(all_labels, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = conf_matrix.tolist()
        
        return total_loss / len(val_loader), metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_dir: Optional[Path] = None) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        # Setup learning rate scheduler
        scheduler = OneCycleLR(
            self.optimizer,
            **TRAINING_CONFIG["scheduler"]["params"],
            steps_per_epoch=len(train_loader),
            epochs=num_epochs
        )
        
        # Setup early stopping
        early_stopping = EarlyStopping()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "learning_rates": []
        }
        
        # Create save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for epoch in range(num_epochs):
                if self.stop_training:
                    print("Training stopped by user. Saving final state...")
                    break
                    
                epoch_start_time = time.time()
                
                # Train
                train_loss = self.train_epoch(train_loader, self.optimizer, epoch + 1)
                history["train_loss"].append(train_loss)
                
                # Evaluate
                val_loss, metrics = self.evaluate(val_loader, "val")
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(metrics)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)
                
                # Log progress
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Time: {epoch_time:.2f}s - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Accuracy: {metrics['accuracy']:.4f} - "
                    f"LR: {current_lr:.2e}"
                )
                
                # Save checkpoint if best model
                if save_dir and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pt"
                    self.save_checkpoint(checkpoint_path, epoch, metrics)
                    logger.info(f"Saved best model checkpoint to {checkpoint_path}")
                
                # Early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
                
                # Learning rate scheduling
                scheduler.step()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            # Save checkpoint here
            if save_dir:
                checkpoint_path = save_dir / "interrupted_checkpoint.pt"
                self.save_checkpoint(checkpoint_path, epoch, {})
                print(f"Saved interrupt checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            # Save final history
            if save_dir:
                history_path = save_dir / "training_history.json"
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=4)
                logger.info(f"Saved training history to {history_path}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return history

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Current metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> int:
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Epoch number of the checkpoint
        """
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["epoch"] 