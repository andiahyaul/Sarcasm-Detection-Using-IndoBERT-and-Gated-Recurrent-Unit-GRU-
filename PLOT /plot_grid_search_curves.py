import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_learning_curves():
    """Plot learning curves from grid search results"""
    # Set style
    plt.style.use('seaborn')
    results_dir = Path('automated_experiments/results')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Grid Search Learning Curves', fontsize=16, y=0.95)
    
    # Plot settings
    scenarios = {
        '70_30_stem': ('70/30 with stemming', 'blue'),
        '70_30_nostem': ('70/30 without stemming', 'green'),
        '80_20_stem': ('80/20 with stemming', 'red'),
        '80_20_nostem': ('80/20 without stemming', 'purple')
    }
    
    # Load and plot results for each scenario
    for file in results_dir.glob('grid_exp_*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            
            # Get scenario info
            split = data['scenario']['split']
            stem = 'stem' if data['scenario']['stemming'] else 'nostem'
            scenario_key = f'{split}_{stem}'
            scenario_name, color = scenarios[scenario_key]
            
            history = data['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Plot training and validation loss
            ax1.plot(epochs, history['train_loss'], '--', color=color, alpha=0.3)
            ax1.plot(epochs, history['val_loss'], '-', color=color, 
                    label=f'{scenario_name}')
            
            # Plot F1-scores
            ax2.plot(epochs, history['train_f1'], '--', color=color, alpha=0.3)
            ax2.plot(epochs, history['val_f1'], '-', color=color, 
                    label=f'{scenario_name}')
            
            # Plot accuracy
            accuracies = [metrics['accuracy'] for metrics in history['val_metrics']]
            ax3.plot(epochs, accuracies, '-', color=color, 
                    label=f'{scenario_name}')
            
            # Plot precision-recall
            precisions = [metrics['weighted avg']['precision'] for metrics in history['val_metrics']]
            recalls = [metrics['weighted avg']['recall'] for metrics in history['val_metrics']]
            ax4.plot(recalls, precisions, '-o', color=color, 
                    label=f'{scenario_name}')
    
    # Customize subplots
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('F1-Score Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('Accuracy Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('Precision-Recall Curves')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend()
    ax4.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('automated_experiments/plots/grid_search_curves.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'automated_experiments/plots/grid_search_curves.png'")

if __name__ == "__main__":
    plot_learning_curves() 