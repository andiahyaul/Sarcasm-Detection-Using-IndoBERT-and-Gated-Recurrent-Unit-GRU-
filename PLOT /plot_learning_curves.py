import json
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves():
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Read experiment results
    with open('automated_experiments/results/exp_70_30_nostem_iter4_h512_d0.3_lr0.001_b32_l1.json', 'r') as f:
        data = json.load(f)
    
    # Extract metrics
    epochs = range(1, len(data['history']['train_loss']) + 1)
    train_loss = data['history']['train_loss']
    val_loss = data['history']['val_loss']
    val_f1 = data['history']['val_f1']
    
    # Extract accuracy values
    val_accuracy = [metrics['accuracy'] for metrics in data['history']['val_metrics']]
    
    # Plot losses
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=12, pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot metrics
    ax2.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.plot(epochs, val_f1, 'm-', label='Validation F1-Score', linewidth=2)
    ax2.set_title('Validation Metrics Progress', fontsize=12, pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add experiment details
    plt.figtext(0.02, 0.02, f"Experiment: {data['experiment_name']}\n" +
                f"Configuration: Hidden Size={data['params']['hidden_size']}, " +
                f"Dropout={data['params']['dropout']}, " +
                f"Learning Rate={data['params']['learning_rate']}, " +
                f"Batch Size={data['params']['batch_size']}, " +
                f"GRU Layers={data['params']['gru_layers']}\n" +
                f"Best F1: {data['best_f1']:.4f}, " +
                f"Final Accuracy: {data['final_metrics']['accuracy']:.4f}",
                fontsize=8, ha='left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('automated_experiments/plots/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_learning_curves()