import logging
import os
from typing import Tuple, Optional

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from src.experiments.config import INDOBERT_CONFIG, TOKENIZER_CONFIG, DATASET_CONFIG, DATALOADER_CONFIG
from src.preprocessing.data_preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_preprocessed_data(mode="with_stemming"):
    """
    Load preprocessed data from CSV file
    
    Args:
        mode (str): "with_stemming" or "without_stemming"
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    file_path = f"data/processed/preprocessed_sarcasm_data_{mode}.csv"
    logger.info(f"Loading data from {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(data, max_length):
    """
    Prepare data for IndoBERT
    
    Args:
        data (pd.DataFrame): Input dataframe
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Dictionary containing tokenized data and labels
    """
    logger.info("Preparing data for IndoBERT")
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_CONFIG["model_name"])
    
    return {
        "data": tokenizer(
            data["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ),
        "labels": torch.tensor(data["label"].tolist())
    }

def split_dataset(data, train_ratio, val_ratio, test_ratio):
    """
    Split dataset into train, validation and test sets
    
    Args:
        data (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        """
        Initialize SarcasmDataset
        
        Args:
            texts (list): List of text samples
            labels (list): List of labels
            max_length (int): Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(INDOBERT_CONFIG["model_name"])
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        logger.info(f"Created dataset with {len(texts)} samples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def setup_data_pipeline(stemming_mode="with_stemming", split_mode="70_30", max_length=512, batch_size=32, subset_size=None):
    """
    Setup complete data pipeline
    
    Args:
        stemming_mode (str): "with_stemming" or "without_stemming"
        split_mode (str): "70_30" or "80_20"
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for dataloaders
        subset_size (int, optional): If provided, use only this many samples for training
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Setting up data pipeline - Mode: {stemming_mode}, Split: {split_mode}, Max Length: {max_length}, Batch Size: {batch_size}")
    
    # Load data
    data = load_preprocessed_data(stemming_mode)
    
    # If subset_size is provided, take a random subset of the data
    if subset_size is not None and subset_size < len(data):
        logger.info(f"Using subset of {subset_size} samples for grid search")
        data = data.sample(n=subset_size, random_state=42).reset_index(drop=True)
    
    # Get split ratios
    split_ratios = DATASET_CONFIG[f"split_{split_mode}"]
    
    # Split data
    train_data, val_data, test_data = split_dataset(
        data,
        split_ratios["train_ratio"],
        split_ratios["val_ratio"],
        split_ratios["test_ratio"]
    )
    
    # Create datasets
    train_dataset = SarcasmDataset(
        train_data["text"].tolist(),
        train_data["label"].tolist(),
        max_length
    )
    
    val_dataset = SarcasmDataset(
        val_data["text"].tolist(),
        val_data["label"].tolist(),
        max_length
    )
    
    test_dataset = SarcasmDataset(
        test_data["text"].tolist(),
        test_data["label"].tolist(),
        max_length
    )
    
    # Create dataloaders with custom batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info("Data pipeline setup completed")
    
    return train_loader, val_loader, test_loader

class TextClassificationDataset(Dataset):
    def __init__(self, texts: list, labels: list, preprocessor: TextPreprocessor):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of input texts
            labels (list): List of corresponding labels
            preprocessor (TextPreprocessor): Instance of TextPreprocessor
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
        assert len(self.texts) == len(self.labels), "Texts and labels must have the same length"
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess text
        processed = self.preprocessor.preprocess_text(text)
        
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long)
        }

class DataManager:
    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initialize the DataManager.
        
        Args:
            preprocessor (TextPreprocessor): Instance of TextPreprocessor
        """
        self.preprocessor = preprocessor
        self.batch_size = INDOBERT_CONFIG["batch_size"]
        
    def load_data(self, data_path: str, text_column: str, 
                 label_column: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path (str): Path to the data file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
            
        # Verify required columns exist
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns {text_column} and/or {label_column} not found in data")
            
        return self.preprocessor.preprocess_dataframe(df, text_column, label_column)
        
    def prepare_data(self, df: pd.DataFrame, 
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training, validation, and testing.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Create dataset
        dataset = TextClassificationDataset(
            texts=df["cleaned_text"].tolist(),
            labels=df["label"].tolist(),
            preprocessor=self.preprocessor
        )
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader
        
    def get_class_weights(self, labels: list) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            labels (list): List of labels
            
        Returns:
            torch.Tensor: Tensor of class weights
        """
        labels = np.array(labels)
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        class_weights = torch.FloatTensor(
            total_samples / (len(class_counts) * class_counts)
        )
        
        return class_weights 