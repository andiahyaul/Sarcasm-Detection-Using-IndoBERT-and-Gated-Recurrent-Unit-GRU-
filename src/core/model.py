import torch
import torch.nn as nn
from transformers import AutoModel
import logging

from src.experiments.config import INDOBERT_CONFIG, GRU_CONFIG

logger = logging.getLogger(__name__)

class SarcasmModel(nn.Module):
    def __init__(self, model_config=INDOBERT_CONFIG, gru_config=GRU_CONFIG):
        """
        Initialize SarcasmModel
        
        Args:
            model_config: IndoBERT configuration
            gru_config: GRU configuration
        """
        super().__init__()
        
        # Load IndoBERT
        logger.info(f"Loading IndoBERT model: {model_config['model_name']}")
        self.bert = AutoModel.from_pretrained(model_config["model_name"])
        
        # Freeze all BERT parameters except the last layer
        logger.info("Freezing BERT parameters except last layer")
        for name, param in self.bert.named_parameters():
            if "layer.11" not in name:  # Only unfreeze the last layer
                param.requires_grad = False
        
        # Simplified GRU
        logger.info(f"Initializing GRU with hidden size: {gru_config['hidden_sizes'][0]}")
        self.gru = nn.GRU(
            input_size=model_config["hidden_size"],
            hidden_size=gru_config["hidden_sizes"][0],
            num_layers=gru_config["num_layers"],
            dropout=gru_config["dropout"],
            bidirectional=gru_config["bidirectional"],
            batch_first=True
        )
        
        # Simplified classification head
        gru_output_size = gru_config["hidden_sizes"][0] * (2 if gru_config["bidirectional"] else 1)
        self.classifier = nn.Linear(gru_output_size, model_config["num_classes"])
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
            
        self.to(self.device)
        
        # Log model architecture
        logger.info(f"Model Architecture:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- IndoBERT hidden size: {model_config['hidden_size']}")
        logger.info(f"- GRU hidden size: {gru_config['hidden_sizes'][0]}")
        logger.info(f"- Number of classes: {model_config['num_classes']}")
        logger.info(f"- Trainable parameters: {self.count_parameters():,}")
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get BERT embeddings
        with torch.set_grad_enabled(False):
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            sequence_output = bert_output.last_hidden_state
        
        # Apply GRU
        gru_output, _ = self.gru(sequence_output)
        
        # Get the last hidden state
        if self.gru.bidirectional:
            # Concatenate forward and backward last hidden states
            last_hidden = torch.cat((gru_output[:, -1, :self.gru.hidden_size],
                                   gru_output[:, 0, self.gru.hidden_size:]), dim=1)
        else:
            last_hidden = gru_output[:, -1, :]
        
        # Classification
        logits = self.classifier(last_hidden)
        
        return logits
    
    def count_parameters(self):
        """
        Count number of trainable parameters
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, input_ids, attention_mask):
        """
        Get attention weights for interpretation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            attention_weights: Attention weights from last layer
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            # Get attention weights from last layer
            attention_weights = outputs.attentions[-1]
        
        return attention_weights 