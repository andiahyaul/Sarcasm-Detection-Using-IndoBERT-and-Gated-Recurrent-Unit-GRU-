# Sarcasm Detection Model

This project implements a sarcasm detection model using IndoBERT and GRU for Indonesian text classification.

## Project Structure

```
├── src/
│   ├── core/              # Core model components (model, dataset, training)
│   ├── experiments/       # Experiment configurations and runners
│   ├── preprocessing/     # Text preprocessing utilities
│   ├── visualization/     # Visualization tools
│   └── utils/             # Utility functions
├── data/
│   ├── raw/               # Raw dataset files
│   └── processed/         # Preprocessed data files
├── models/                # Saved model checkpoints
├── results/               # Experiment results and reports
├── docs/                  # Documentation
├── plots/                 # Generated plots
└── requirements.txt       # Project dependencies
```

## Dependencies

- torch>=1.9.0
- transformers>=4.12.0
- pandas>=1.3.0
- numpy>=1.19.5
- scikit-learn>=0.24.2
- tqdm>=4.62.3
- Sastrawi>=1.0.1
- matplotlib>=3.4.3
- seaborn>=0.11.2
- plotly>=5.3.1

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run all experiments:
```bash
python src/experiments/main.py --mode all
```

Run a single experiment:
```bash
python src/experiments/main.py --mode single --stemming with_stemming --split 80_20 --dimension 768
```

## Components

- **Model**: Uses IndoBERT with GRU for sequence classification
- **Preprocessing**: Text cleaning, normalization, and optional stemming
- **Training**: Configurable training pipeline with early stopping
- **Evaluation**: Comprehensive model evaluation with metrics and visualizations