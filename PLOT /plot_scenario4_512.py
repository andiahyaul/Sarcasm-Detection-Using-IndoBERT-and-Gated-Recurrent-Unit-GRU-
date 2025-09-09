import json
import matplotlib.pyplot as plt
import numpy as np

def plot_scenario4_512_experiments():
    """Plot training progress for both Scenario 4 experiments with hidden size 512"""
    # List of experiments to plot
    experiments = [
        ('exp_80_20_nostem_iter0_h512_d0.1_lr0.0001_b32_l2.json', 1, "Dropout 0.1"),
        ('exp_80_20_nostem_iter1_h512_d0.2_lr0.0001_b32_l2.json', 2, "Dropout 0.2")
    ]

    plt.figure(figsize=(15, 6))

    # 1. Training Progress
    plt.subplot(1, 2, 1)

    for file_name, exp_num, label in experiments:
        try:
            # Load training history
            history_path = f'automated_experiments/results/{file_name}'
            with open(history_path, 'r') as f:
                results = json.load(f)

            history = results['history']
            epochs = range(1, len(history['val_f1']) + 1)
            f1_scores = history['val_f1']

            # Plot F1-scores with value labels
            plt.plot(epochs, f1_scores, marker='o', label=label, linewidth=2, markersize=8)
            for i, score in enumerate(f1_scores, 1):
                plt.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')

        except FileNotFoundError:
            print(f"Results file not found for experiment {exp_num}")
            continue

    plt.title('Scenario 4 - Hidden Size 512 Progress', fontsize=12, pad=15)
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

    # Create text for display
    param_text = "Experiment Parameters:\n\n"
    param_text += "Common Parameters:\n"
    param_text += "Hidden Size: 512\n"
    param_text += "Learning Rate: 0.0001\n"
    param_text += "Batch Size: 32\n"
    param_text += "GRU Layers: 2\n\n"

    for file_name, exp_num, label in experiments:
        try:
            with open(f'automated_experiments/results/{file_name}', 'r') as f:
                results = json.load(f)
            
            final_metrics = results['history']['val_metrics'][-1]
            param_text += f"\n{label} Results:\n"
            param_text += f"Best F1-Score: {results.get('best_f1', max(results['history']['val_f1'])):.4f}\n"
            param_text += f"Final Metrics:\n"
            param_text += f"Accuracy: {final_metrics['accuracy']:.4f}\n"
            param_text += f"F1-Score: {final_metrics['weighted avg']['f1-score']:.4f}\n"
            param_text += f"Precision: {final_metrics['weighted avg']['precision']:.4f}\n"
            param_text += f"Recall: {final_metrics['weighted avg']['recall']:.4f}\n"
        except FileNotFoundError:
            param_text += f"\n{label}: Results not available yet\n"

    plt.text(0.5, 0.5, param_text, 
             fontsize=12, 
             va='center', 
             ha='center',
             bbox=dict(facecolor='white', 
                      alpha=0.8, 
                      edgecolor='gray', 
                      boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig('automated_experiments/plots/scenario4_512_experiments.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'automated_experiments/plots/scenario4_512_experiments.png'")

if __name__ == "__main__":
    plot_scenario4_512_experiments() 