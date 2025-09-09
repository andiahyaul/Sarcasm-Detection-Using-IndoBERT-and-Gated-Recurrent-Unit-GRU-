import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory"""
    results = []
    for file in Path(results_dir).glob('grid_exp_*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            # Extract scenario and parameters
            scenario = data['scenario']
            params = data['params']
            # Get best F1 score and final metrics
            best_f1 = data['best_f1']
            final_metrics = data['history']['val_metrics'][-1]
            
            results.append({
                'split': scenario['split'],
                'stemming': scenario['stemming'],
                'hidden_size': params['hidden_size'],
                'dropout': params['dropout'],
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'gru_layers': params['gru_layers'],
                'best_f1': best_f1,
                'final_accuracy': final_metrics['accuracy'],
                'final_f1': final_metrics['weighted avg']['f1-score'],
                'final_precision': final_metrics['weighted avg']['precision'],
                'final_recall': final_metrics['weighted avg']['recall'],
                'history': data['history']
            })
    return pd.DataFrame(results)

def plot_parameter_comparison(df, save_dir):
    """Create plots comparing performance across different parameter values"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create parameter comparison plots
    params = ['hidden_size', 'dropout', 'learning_rate', 'batch_size', 'gru_layers']
    metrics = ['best_f1', 'final_accuracy', 'final_precision', 'final_recall']
    
    for param in params:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Impact of {param.replace("_", " ").title()} on Model Performance', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            sns.boxplot(x=param, y=metric, data=df, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel(metric.replace("_", " ").title())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{param}_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_scenario_comparison(df, save_dir):
    """Create plots comparing performance across different scenarios"""
    sns.set_style("whitegrid")
    
    # Prepare data for plotting
    scenarios = []
    for _, row in df.iterrows():
        scenario_name = f"{row['split']} {'with' if row['stemming'] else 'without'} stemming"
        scenarios.append(scenario_name)
    df['scenario'] = scenarios
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Comparison Across Scenarios', fontsize=16)
    
    metrics = {
        (0,0): ('best_f1', 'Best F1 Score'),
        (0,1): ('final_accuracy', 'Final Accuracy'),
        (1,0): ('final_precision', 'Final Precision'),
        (1,1): ('final_recall', 'Final Recall')
    }
    
    for (i,j), (metric, title) in metrics.items():
        sns.boxplot(x='scenario', y=metric, data=df, ax=axes[i,j])
        axes[i,j].set_title(title)
        axes[i,j].set_xlabel('Scenario')
        axes[i,j].set_ylabel(title)
        plt.setp(axes[i,j].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(df, save_dir):
    """Plot learning curves for each scenario"""
    sns.set_style("whitegrid")
    scenarios = df['scenario'].unique()
    
    for scenario in scenarios:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Learning Curves - {scenario}', fontsize=16)
        
        # Get experiments for this scenario
        scenario_df = df[df['scenario'] == scenario]
        
        # Plot training and validation loss
        for _, row in scenario_df.iterrows():
            history = row['history']
            config = f'h={row["hidden_size"]}, d={row["dropout"]}'
            
            ax1.plot(history['train_loss'], '--', label=f'Train ({config})')
            ax1.plot(history['val_loss'], '-', label=f'Val ({config})')
        
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # Plot F1 scores
        for _, row in scenario_df.iterrows():
            history = row['history']
            config = f'h={row["hidden_size"]}, d={row["dropout"]}'
            
            ax2.plot(history['train_f1'], '--', label=f'Train ({config})')
            ax2.plot(history['val_f1'], '-', label=f'Val ({config})')
        
        ax2.set_title('F1 Score Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'learning_curves_{scenario.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_table(df):
    """Create a summary table of the best performing configurations"""
    # Get best configuration for each scenario
    summary = []
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        best_row = scenario_df.loc[scenario_df['best_f1'].idxmax()]
        summary.append({
            'Scenario': scenario,
            'Best F1': f"{best_row['best_f1']:.4f}",
            'Hidden Size': best_row['hidden_size'],
            'Dropout': best_row['dropout'],
            'Learning Rate': f"{best_row['learning_rate']:.6f}",
            'Batch Size': best_row['batch_size'],
            'GRU Layers': best_row['gru_layers']
        })
    return pd.DataFrame(summary)

def main():
    # Set paths
    results_dir = Path('grid_search_experiments/results')
    plots_dir = Path('grid_search_experiments/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = load_experiment_results(results_dir)
    
    # Create plots
    plot_parameter_comparison(df, plots_dir)
    plot_scenario_comparison(df, plots_dir)
    plot_learning_curves(df, plots_dir)
    
    # Create and save summary table
    summary_df = create_summary_table(df)
    summary_df.to_csv(plots_dir / 'best_configurations.csv', index=False)
    
    print("\nPlots have been saved to:", plots_dir)
    print("\nBest Configurations:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main() 