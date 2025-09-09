import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_metrics_progress():
    """Plot metrics progress focusing only on F1-scores"""
    print("Starting plot generation...")
    results_dir = Path('automated_experiments/results')
    
    # Target files with highest F1-scores
    target_files = [
        'exp_70_30_nostem_iter4_h512_d0.3_lr0.001_b32_l1.json',  # Best 70/30 without stemming
        'exp_80_20_nostem_iter2_h256_d0.2_lr0.0001_b16_l2.json',  # Best 80/20 without stemming
        'exp_70_30_stem_iter0_h256_d0.2_lr0.001_b32_l1.json',  # Best 70/30 with stemming
        'exp_80_20_stem_iter1_h256_d0.1_lr0.0003_b32_l1.json'  # Best 80/20 with stemming
    ]
    
    for file_path in target_files:
        full_path = results_dir / file_path
        if not full_path.exists():
            continue
            
        print(f"Processing file: {file_path}")
        with open(full_path, 'r') as f:
            data = json.load(f)
            
            # Get metrics
            f1_scores = data['history']['val_f1']
            
            # Create figure with 2 subplots side by side
            plt.figure(figsize=(15, 6))
            
            # 1. F1-Score Progress
            plt.subplot(1, 2, 1)
            epochs = range(1, len(f1_scores) + 1)
            
            # Plot F1-scores
            plt.plot(epochs, f1_scores, '-o', color='orange', label='F1-Score', linewidth=2, markersize=8)
            
            # Add value labels
            for i, f1 in enumerate(f1_scores, 1):
                plt.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', va='bottom')
            
            # Get scenario info from filename
            name_parts = file_path.split('_')
            split = f"{name_parts[1]}/{name_parts[2]}"
            stem_text = "with" if name_parts[3] == "stem" else "without"
            plt.title(f'{split} {stem_text} stemming - F1-Score Progress', fontsize=12, pad=15)
            
            plt.xlabel('Epoch')
            plt.ylabel('F1-Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Set y-axis limits to show differences clearly
            plt.ylim(0.70, 0.85)
            
            # 2. Configuration and Final Metrics
            plt.subplot(1, 2, 2)
            plt.axis('off')
            
            # Get configuration from data
            config = data['params']
            
            # Create text for display
            param_text = "Experiment Parameters:\n\n"
            param_text += f"Hidden Size: {config['hidden_size']}\n"
            param_text += f"Dropout: {config['dropout']}\n"
            param_text += f"Learning Rate: {config['learning_rate']}\n"
            param_text += f"Batch Size: {config['batch_size']}\n"
            param_text += f"GRU Layers: {config['gru_layers']}\n\n"
            
            param_text += "F1-Score Metrics:\n"
            param_text += f"Best F1-Score: {data['best_f1']:.4f}\n"
            param_text += f"Final F1-Score: {f1_scores[-1]:.4f}\n\n"
            
            param_text += "Progress by Epoch:\n"
            for epoch, f1 in enumerate(f1_scores, 1):
                param_text += f"Epoch {epoch}: {f1:.4f}\n"
            
            plt.text(0.5, 0.5, param_text,
                    fontsize=12,
                    va='center',
                    ha='center',
                    bbox=dict(facecolor='white',
                             alpha=0.8,
                             edgecolor='gray',
                             boxstyle='round,pad=1'))
            
            plt.tight_layout()
            scenario_name = f"{name_parts[1]}_{name_parts[2]}_{name_parts[3]}"
            save_path = Path(f'automated_experiments/plots/{scenario_name}_f1_progress.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as '{save_path}'")
            
            # Print results
            print(f"\nResults for {scenario_name}:")
            print(f"Best F1-Score: {data['best_f1']:.4f}")
            print("Configuration:", config)
            plt.close()

if __name__ == "__main__":
    plot_metrics_progress() 