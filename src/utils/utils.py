import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from src.experiments.config import LOG_CONFIG

# Setup logging
logging.basicConfig(
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_model(model: nn.Module, save_path: Path, metadata: Optional[Dict[str, Any]] = None):
    """
    Save model and metadata.
    
    Args:
        model (nn.Module): Model to save
        save_path (Path): Path to save model
        metadata (Dict[str, Any], optional): Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    
    # Save model
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to {save_path}")
    
def load_model(model: nn.Module, load_path: Path) -> Dict[str, Any]:
    """
    Load model and metadata.
    
    Args:
        model (nn.Module): Model to load weights into
        load_path (Path): Path to load model from
        
    Returns:
        Dict[str, Any]: Model metadata
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found at {load_path}")
        
    # Load model
    save_dict = torch.load(load_path)
    model.load_state_dict(save_dict["model_state_dict"])
    logger.info(f"Model loaded from {load_path}")
    
    return save_dict.get("metadata", {})
    
def save_config(config: Dict[str, Any], save_path: Path):
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        save_path (Path): Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {save_path}")
    
def load_config(load_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        load_path (Path): Path to load configuration from
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {load_path}")
        
    with open(load_path, "r") as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {load_path}")
    
    return config
    
def setup_logging(log_file: Optional[Path] = None):
    """
    Setup logging configuration.
    
    Args:
        log_file (Path, optional): Path to log file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"],
        handlers=handlers
    )
    
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count number of trainable and non-trainable parameters.
    
    Args:
        model (nn.Module): Model to count parameters for
        
    Returns:
        Dict[str, int]: Dictionary containing parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total": total_params
    }
    
def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")
    
def get_device() -> torch.device:
    """
    Get device for training.
    
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device 