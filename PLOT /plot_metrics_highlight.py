import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_metrics_highlight():
    """Plot F1-scores with highlighting for results ≥ 80%"""
    print("Starting plot generation...")
    results_dir = Path('automated_experiments/results')
    print(f"Looking for results in: {results_dir}")
    
    # Storage for highest metrics
    highest_metrics = {
        '70_30_stem': {'f1': 0, 'config': None},
        '70_30_nostem': {'f1': 0, 'config': None},
        '80_20_stem': {'f1': 0, 'config': None},
        '80_20_nostem': {'f1': 0, 'config': None}
    }
    
    # Load and process results
    result_files = list(results_dir.glob('exp_*_*_*.json'))
    print(f"Found {len(result_files)} result files")
    
    for file in result_files:
        if file.name == 'final_report.json':
            continue
            
        try:
            print(f"Processing file: {file.name}")
            with open(file, 'r') as f:
                data = json.load(f)
                
                # Get scenario info from filename
                name_parts = file.stem.split('_')
                split = f"{name_parts[1]}_{name_parts[2]}"
                stem_part = name_parts[3]
                scenario_key = f'{split}_{stem_part}'
                
                # Get F1-scores
                history = data['history']
                f1_scores = [metrics['weighted avg']['f1-score'] for metrics in history['val_metrics']]
                max_f1 = max(f1_scores)
                print(f"  {scenario_key} - Max F1: {max_f1:.4f}")
                
                # Parse configuration
                config = {
                    'hidden_size': int(name_parts[5][1:]),
                    'dropout': float(name_parts[6][1:]),
                    'learning_rate': float(name_parts[7][2:]),
                    'batch_size': int(name_parts[8][1:]),
                    'gru_layers': int(name_parts[9][1:])
                }
                
                # Update if better F1 found
                if max_f1 > highest_metrics[scenario_key]['f1']:
                    highest_metrics[scenario_key]['f1'] = max_f1
                    highest_metrics[scenario_key]['config'] = config
                    
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            continue
    
    print("\nBest F1-scores found:")
    for scenario, metrics in highest_metrics.items():
        print(f"{scenario}: F1={metrics['f1']:.4f}")
    
    print("\nGenerating plot...")
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot settings
    scenario_names = [
        '70/30 with stemming',
        '70/30 without stemming',
        '80/20 with stemming',
        '80/20 without stemming'
    ]
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot F1-scores
    f1_scores = [metrics['f1'] for metrics in highest_metrics.values()]
    bars = plt.bar(scenario_names, f1_scores, color=colors)
    plt.title('Highest F1-Scores Achieved\nHighlighting Results ≥ 80%', fontsize=14)
    plt.ylabel('F1-Score')
    plt.tick_params(axis='x', rotation=45)
    plt.ylim(0.70, 0.85)
    plt.grid(True, alpha=0.3)
    
    # Add 80% line
    plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    plt.legend()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        color = 'darkred' if height >= 0.80 else 'black'
        weight = 'bold' if height >= 0.80 else 'normal'
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom',
                color=color,
                weight=weight)
    
    # Add configuration details
    config_text = "Best Configurations:\n\n"
    for scenario_key, metrics in highest_metrics.items():
        scenario_name = scenario_key.replace('_', '/')
        config_text += f"{scenario_name}:\n"
        
        # Highlight F1-scores ≥ 80%
        f1_text = f"{metrics['f1']:.4f}"
        if metrics['f1'] >= 0.80:
            f1_text = f"★ {f1_text} ★"
            
        config_text += f"F1-Score: {f1_text}\n"
        
        if metrics['config']:
            config = metrics['config']
            config_text += f"Hidden Size: {config['hidden_size']}, "
            config_text += f"Dropout: {config['dropout']}, "
            config_text += f"LR: {config['learning_rate']}\n"
            config_text += f"Batch Size: {config['batch_size']}, "
            config_text += f"GRU Layers: {config['gru_layers']}\n"
        config_text += "\n"
    
    plt.figtext(0.5, -0.4, config_text, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    print("Saving plot...")
    plt.tight_layout()
    save_path = Path('automated_experiments/plots/f1_scores_highlight.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as '{save_path}'")
    
    # Print detailed results
    print("\nDetailed F1-Score Analysis:")
    for scenario_key, metrics in highest_metrics.items():
        print(f"\n{scenario_key}:")
        f1 = metrics['f1']
        f1_text = f"{f1:.4f}"
        if f1 >= 0.80:
            f1_text = f"★ {f1_text} ★"
        print(f"F1-Score: {f1_text}")
        if metrics['config']:
            print("Best Configuration:", metrics['config'])

if __name__ == "__main__":
    plot_metrics_highlight()