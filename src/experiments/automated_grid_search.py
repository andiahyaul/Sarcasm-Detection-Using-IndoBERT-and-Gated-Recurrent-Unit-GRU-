import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
from src.experiments.automated_config import (
    DEVICE, MODEL_CONFIG, SCENARIOS, TRAINING_CONFIG,
    PARAMETER_GRID, COMBINATIONS_PER_SCENARIO,
    EXPERIMENT_DIR, RESULTS_DIR, generate_grid_combinations,
    get_experiment_name
)

class SarcasmDataset:
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Convert texts to list of strings if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        # Ensure all texts are strings
        texts = [str(text) for text in texts]
        
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class SarcasmModel(nn.Module):
    def __init__(self, bert_model, hidden_size, dropout, num_classes, gru_layers):
        super().__init__()
        self.bert = bert_model
        self.gru = nn.GRU(bert_model.config.hidden_size, hidden_size, 
                         num_layers=gru_layers, batch_first=True, 
                         bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        pooled_output = torch.mean(gru_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class GridSearchTrainer:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        self.bert_model = AutoModel.from_pretrained(MODEL_CONFIG["model_name"])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(EXPERIMENT_DIR / 'grid_search.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Freeze BERT parameters except last layer
        for param in list(self.bert_model.parameters())[:-2]:
            param.requires_grad = False
            
        self.bert_model = self.bert_model.to(self.device)
        
    def prepare_data(self, scenario):
        file_suffix = "with_stemming" if scenario["stemming"] else "without_stemming"
        data_path = f"data/processed/preprocessed_sarcasm_data_{file_suffix}.csv"
        
        # Load and split data
        df = pd.read_csv(data_path)
        split_ratio = float(scenario["split"].split("_")[0]) / 100
        
        # Convert texts to list of strings
        texts = df['text'].tolist()
        labels = df['label'].values
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            train_size=split_ratio,
            random_state=42
        )
        
        # Create datasets
        train_dataset = SarcasmDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SarcasmDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
        
    def train_model(self, model, train_loader, val_loader, optimizer, params):
        best_f1 = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'val_metrics': []
        }
        
        for epoch in range(TRAINING_CONFIG["num_epochs"]):
            # Training
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{TRAINING_CONFIG["num_epochs"]}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            TRAINING_CONFIG["gradient_clipping"])
                optimizer.step()
                
                total_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = total_loss / len(train_loader)
            train_metrics = classification_report(train_labels, train_preds, 
                                               output_dict=True)
            train_f1 = train_metrics['weighted avg']['f1-score']
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_metrics = classification_report(val_labels, val_preds, 
                                             output_dict=True)
            val_f1 = val_metrics['weighted avg']['f1-score']
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            history['val_metrics'].append(val_metrics)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
            
            self.logger.info(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']}")
            self.logger.info(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            self.logger.info(f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        return history, best_f1
    
    def run_grid_search(self):
        start_time = time.time()
        self.logger.info("Starting Grid Search Training")
        self.logger.info(f"Device: {DEVICE}")
        self.logger.info(f"Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        grid_combinations = generate_grid_combinations()
        
        for scenario_idx, scenario in enumerate(SCENARIOS):
            self.logger.info(f"\nStarting Scenario {scenario_idx + 1}")
            self.logger.info(f"Split: {scenario['split']}, Stemming: {scenario['stemming']}")
            
            train_dataset, val_dataset = self.prepare_data(scenario)
            
            for iteration, params in enumerate(grid_combinations):
                exp_name = get_experiment_name(scenario_idx, iteration, params)
                self.logger.info(f"\nStarting Experiment: {exp_name}")
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, 
                                       batch_size=params['batch_size'],
                                       shuffle=True)
                val_loader = DataLoader(val_dataset,
                                      batch_size=params['batch_size'],
                                      shuffle=False)
                
                # Initialize model
                model = SarcasmModel(
                    self.bert_model,
                    params['hidden_size'],
                    params['dropout'],
                    MODEL_CONFIG['num_classes'],
                    params['gru_layers']
                ).to(self.device)
                
                # Initialize optimizer
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=TRAINING_CONFIG['weight_decay']
                )
                
                # Train model
                history, best_f1 = self.train_model(model, train_loader, 
                                                  val_loader, optimizer, params)
                
                # Save results
                results = {
                    'params': params,
                    'history': history,
                    'best_f1': best_f1,
                    'scenario': {
                        'split': scenario['split'],
                        'stemming': scenario['stemming']
                    }
                }
                
                with open(RESULTS_DIR / f"{exp_name}.json", 'w') as f:
                    json.dump(results, f, indent=4)
                
                self.logger.info(f"Completed Experiment: {exp_name}")
                self.logger.info(f"Best F1 Score: {best_f1:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"\nGrid Search completed in {total_time / 3600:.2f} hours")

if __name__ == "__main__":
    trainer = GridSearchTrainer()
    trainer.run_grid_search() 