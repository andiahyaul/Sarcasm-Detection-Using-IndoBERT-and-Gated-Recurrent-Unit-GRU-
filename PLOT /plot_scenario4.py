import json
import matplotlib.pyplot as plt
import numpy as np

def plot_scenario4_progress():
    """Plot training progress for scenario 4 (80-20 split, without stemming)"""
    # Load training history
    history_path = 'models/saved_models/split_80-20_without_stemming_dim_768/training_history.json'
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

    plt.title('Scenario 4 - F1-Score Progress (80-20 split, without stemming)', fontsize=12, pad=15)
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
    
    # Parameters used in training
    params = {
        "Hidden Size": 768,
        "Dropout": 0.1,
        "Learning Rate": "3e-5",
        "Batch Size": 32
    }

    # Create text for display
    param_text = "Scenario 4 Best Parameters:\n\n"
    for param, value in params.items():
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
    plt.savefig('plots/scenario_4_progress.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'plots/scenario_4_progress.png'")

if __name__ == "__main__":
    plot_scenario4_progress() 