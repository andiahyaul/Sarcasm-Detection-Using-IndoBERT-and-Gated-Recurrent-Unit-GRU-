import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_training_progress(scenario_num, split_ratio="70_30"):
    """Plot training progress for a specific scenario
    
    Args:
        scenario_num: Scenario number (1 or 2)
        split_ratio: Split ratio ("70_30" or "80_20")
    """
    # Convert split ratio format for filename (e.g., "70_30" to "70-30")
    split_name = split_ratio.replace("_", "-")
    
    # Load training history
    history_path = f'models/saved_models/split_{split_name}_with_stemming_dim_768/training_history.json'
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Create figure with 2 subplots side by side
    plt.figure(figsize=(15, 6))

    # 1. Training Progress
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)

    # Get F1-scores for each epoch
    f1_scores = [metrics['weighted avg']['f1-score'] for metrics in history['val_metrics']]

    # Plot F1-scores with value labels
    plt.plot(epochs, f1_scores, 'r-o', label='F1-Score', linewidth=2, markersize=8)
    for i, score in enumerate(f1_scores, 1):
        plt.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')

    plt.title(f'Scenario {scenario_num} - F1-Score Progress ({split_ratio} split)', fontsize=12, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Adjust y-axis to make room for labels
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax + 0.05)

    # 2. Best Parameters and Final Metrics
    plt.subplot(1, 2, 2)
    plt.axis('off')

    # Get final metrics
    final_metrics = history['val_metrics'][-1]
    best_params = {
        "Hidden Size": 768,
        "Dropout": 0.2,
        "Learning Rate": "3e-5",
        "Batch Size": 32
    }

    # Create text for display
    param_text = f"Scenario {scenario_num} Best Parameters:\n\n"
    for param, value in best_params.items():
        param_text += f"{param}: {value}\n"

    param_text += f"\nF1-Scores by Epoch:\n"
    for epoch, score in enumerate(f1_scores, 1):
        param_text += f"Epoch {epoch}: {score:.4f}\n"

    param_text += f"\nFinal Metrics:\n"
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
    plt.savefig(f'plots/scenario_{scenario_num}_progress.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'plots/scenario_{scenario_num}_progress.png'")

if __name__ == "__main__":
    # If no arguments provided, default to scenario 1
    scenario = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    split = sys.argv[2] if len(sys.argv) > 2 else "70_30"
    
    plot_training_progress(scenario, split) 