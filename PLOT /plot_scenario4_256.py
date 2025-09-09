import json
import matplotlib.pyplot as plt
import numpy as np

def plot_scenario4_256_experiment():
    """Plot training progress for Scenario 4 experiment with hidden size 256"""
    # Load training history
    history_path = 'automated_experiments/results/exp_80_20_nostem_iter0_h256_d0.2_lr0.0001_b32_l2.json'
    with open(history_path, 'r') as f:
        results = json.load(f)

    history = results['history']

    # Create figure with 2 subplots side by side
    plt.figure(figsize=(15, 6))

    # 1. Training Progress
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)

    # Get F1-scores for each epoch
    f1_scores = history['val_f1']

    # Plot F1-scores with value labels
    plt.plot(epochs, f1_scores, 'b-o', label='F1-Score', linewidth=2, markersize=8)
    for i, score in enumerate(f1_scores, 1):
        plt.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')

    plt.title('Scenario 4 - Hidden Size 256 Progress', fontsize=12, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Adjust y-axis to make room for labels
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax + 0.05)

    # 2. Parameters and Final Metrics
    plt.subplot(1, 2, 2)
    plt.axis('off')

    # Get final metrics
    final_metrics = history['val_metrics'][-1]
    params = results['params']

    # Create text for display
    param_text = "Experiment Parameters:\n\n"
    param_text += f"Hidden Size: {params['hidden_size']}\n"
    param_text += f"Dropout: {params['dropout']}\n"
    param_text += f"Learning Rate: {params['learning_rate']}\n"
    param_text += f"Batch Size: {params['batch_size']}\n"
    param_text += f"GRU Layers: {params['gru_layers']}\n"

    param_text += f"\nF1-Scores by Epoch:\n"
    for epoch, score in enumerate(f1_scores, 1):
        param_text += f"Epoch {epoch}: {score:.4f}\n"

    param_text += f"\nBest F1-Score: {results.get('best_f1', max(f1_scores)):.4f}\n"
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
    plt.savefig('automated_experiments/plots/scenario4_256_experiment.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'automated_experiments/plots/scenario4_256_experiment.png'")

if __name__ == "__main__":
    plot_scenario4_256_experiment() 