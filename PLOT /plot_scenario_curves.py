import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_scenario_curves():
    """Plot learning curves for each scenario separately"""
    # Set style
    sns.set_style("whitegrid")
    results_dir = Path('models/saved_models')
    
    # Target files that we know have 80% accuracy
    target_files = [
        'split_80-20_without_stemming_dim_768/training_history.json',
        'split_70-30_with_stemming_dim_768/training_history.json'
    ]
    
    for file_path in target_files:
        full_path = results_dir / file_path
        if not full_path.exists():
            continue
            
        print(f"Processing file: {file_path}")
        with open(full_path, 'r') as f:
            data = json.load(f)
            
            # Get metrics
            accuracies = [metrics['accuracy'] for metrics in data['val_metrics']]
            f1_scores = [metrics['weighted avg']['f1-score'] for metrics in data['val_metrics']]
            
            # Create figure with 2 subplots side by side
            plt.figure(figsize=(15, 6))
            
            # 1. Metrics Progress
            plt.subplot(1, 2, 1)
            epochs = range(1, len(accuracies) + 1)
            
            # Plot accuracy
            plt.plot(epochs, accuracies, '-o', color='green', label='Accuracy', linewidth=2, markersize=6)
            
            # Plot F1-scores
            plt.plot(epochs, f1_scores, '--o', color='orange', label='F1-Score', linewidth=2, markersize=6)
            
            # Add value labels
            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores), 1):
                plt.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom')
                plt.text(i, f1 - 0.005, f'{f1:.4f}', ha='center', va='top')
            
            # Get scenario info from path
            scenario = file_path.split('/')[0]
            split_ratio = scenario.split('_')[1].replace('-', '/')
            stem_text = "with" if "with_stemming" in scenario else "without"
            plt.title(f'{split_ratio} {stem_text} stemming - Metrics Progress', fontsize=12, pad=15)
            
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Set y-axis limits to show differences clearly
            plt.ylim(0.70, 0.85)
            
            # 2. Configuration and Final Metrics
            plt.subplot(1, 2, 2)
            plt.axis('off')
            
            # Configuration for this experiment
            config = {
                'hidden_size': 768,
                'dropout': 0.2 if "with_stemming" in scenario else 0.1,
                'learning_rate': 0.001,
                'batch_size': 32,
                'gru_layers': 1
            }
            
            final_metrics = data['val_metrics'][-1]
            
            # Create text for display
            param_text = "Experiment Parameters:\n\n"
            param_text += f"Hidden Size: {config['hidden_size']}\n"
            param_text += f"Dropout: {config['dropout']}\n"
            param_text += f"Learning Rate: {config['learning_rate']}\n"
            param_text += f"Batch Size: {config['batch_size']}\n"
            param_text += f"GRU Layers: {config['gru_layers']}\n\n"
            
            param_text += "Best Metrics:\n"
            param_text += f"Best Accuracy: {max(accuracies):.4f}\n"
            param_text += f"Best F1-Score: {max(f1_scores):.4f}\n\n"
            
            param_text += "Final Metrics:\n"
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
            save_name = scenario.replace('-', '_')
            save_path = Path(f'automated_experiments/plots/{save_name}_progress.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as '{save_path}'")
            
            # Print results
            print(f"\nResults for {scenario}:")
            print(f"Best Accuracy: {max(accuracies):.4f}")
            print(f"Best F1-Score: {max(f1_scores):.4f}")
            print("Configuration:", config)
            plt.close()

if __name__ == "__main__":
    plot_scenario_curves() 