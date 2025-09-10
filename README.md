# 🚀 Sarcasm Detection Using IndoBERT and GRU

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow)](https://github.com/andiahyaul/Sarcasm-Detection-Using-IndoBERT-and-Gated-Recurrent-Unit-GRU-)

> A sophisticated deep learning hybrid integrating **IndoBERT** and **Gated Recurrent Unit (GRU)** to enhance sarcasm detection accuracy in Indonesian text. The model leverages contextual understanding and sequential processing to effectively identify sarcastic comments on social media.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Sarcasm detection in Indonesian text presents unique challenges due to:
- **Complex linguistic structures** and cultural nuances
- **Limited labeled datasets** for Indonesian language
- **Contextual dependency** requiring deep semantic understanding

This project addresses these challenges by combining:
- **IndoBERT**: Pre-trained Indonesian BERT model for contextual embeddings
- **GRU Networks**: Sequential processing for capturing temporal dependencies
- **Hybrid Architecture**: Leveraging strengths of both transformer and recurrent models

## ✨ Features

- 🧠 **Advanced NLP Pipeline**: Complete preprocessing with Indonesian text normalization
- 🔄 **Hybrid Architecture**: IndoBERT + GRU for optimal performance
- 📊 **Comprehensive Evaluation**: Multiple metrics and visualization tools
- ⚡ **Efficient Training**: Optimized for GPU acceleration (CUDA/MPS)
- 🎯 **Automated Experiments**: Grid search and hyperparameter optimization
- 📈 **Rich Visualizations**: Performance plots and analysis tools

## 🏗️ Model Architecture

```
Input Text
    ↓
[Text Preprocessing]
    ↓
[IndoBERT Tokenization]
    ↓
[IndoBERT Encoder] → [768-dim embeddings]
    ↓
[Bidirectional GRU] → [Hidden representations]
    ↓
[Classification Head] → [Binary Output: Sarcastic/Non-sarcastic]
```

### Key Components:
- **IndoBERT**: Fine-tuned Indonesian BERT (`indobenchmark/indobert-base-p1`)
- **GRU Layer**: Bidirectional processing with configurable hidden sizes
- **Classification Head**: Linear layer with dropout for final prediction

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/andiahyaul/Sarcasm-Detection-Using-IndoBERT-and-Gated-Recurrent-Unit-GRU-.git
   cd Sarcasm-Detection-Using-IndoBERT-and-Gated-Recurrent-Unit-GRU-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   python -c "from transformers import AutoModel; print('Transformers installed successfully')"
   ```

## 💻 Usage

### Quick Start

1. **Prepare your data**
   - Place your dataset in `data/` directory
   - Ensure CSV format with columns: `text`, `label` (0: non-sarcastic, 1: sarcastic)

2. **Run basic training**
   ```bash
   python src/experiments/main.py --split_ratio 80 --use_stemming --epochs 10
   ```

3. **Automated experiments**
   ```bash
   python src/experiments/automated_training.py
   ```

### Advanced Usage

#### Custom Configuration
```python
from src.experiments.config import INDOBERT_CONFIG, GRU_CONFIG

# Modify model parameters
INDOBERT_CONFIG['max_length'] = 256
GRU_CONFIG['hidden_sizes'] = [512, 256]

# Run experiment
from src.experiments.experiment import ExperimentRunner
runner = ExperimentRunner()
runner.run_experiment(split_ratio=80, use_stemming=True)
```

#### Evaluation and Visualization
```python
from src.core.evaluate import evaluate_model
from src.visualization.visualization import plot_training_curves

# Evaluate model
results = evaluate_model(model, test_loader)

# Generate plots
plot_training_curves(training_history, save_path='plots/')
```

## 📁 Project Structure

```
├── 📂 src/
│   ├── 📂 core/           # Core model components
│   │   ├── model.py       # SarcasmModel implementation
│   │   ├── train.py       # Training logic
│   │   ├── evaluate.py    # Evaluation metrics
│   │   └── dataset.py     # Data loading utilities
│   ├── 📂 experiments/    # Experiment configurations
│   │   ├── main.py        # Main experiment runner
│   │   ├── config.py      # Model configurations
│   │   └── experiment.py  # Experiment orchestration
│   ├── 📂 preprocessing/  # Data preprocessing
│   │   └── data_preprocessing.py
│   ├── 📂 visualization/  # Plotting utilities
│   │   └── visualization.py
│   └── 📂 utils/          # Helper functions
│       └── utils.py
├── 📂 data/              # Datasets
├── 📂 models/            # Saved model checkpoints
├── 📂 plots/             # Generated visualizations
├── 📂 results/           # Experiment results
├── 📄 requirements.txt   # Dependencies
└── 📄 README.md         # This file
```

## 🧪 Experiments

### Available Experiments

1. **Basic Training**
   - Single model training with specified parameters
   - Support for different data splits (70-30, 80-20)
   - With/without stemming preprocessing

2. **Grid Search**
   - Automated hyperparameter optimization
   - Multiple GRU hidden sizes: [128, 256, 512]
   - Various dropout rates: [0.1, 0.2, 0.3]
   - Learning rates: [1e-3, 3e-4, 1e-4]

3. **Cross-Validation**
   - K-fold validation for robust evaluation
   - Statistical significance testing
   - Performance consistency analysis

### Running Experiments

```bash
# Basic experiment
python src/experiments/main.py \
    --split_ratio 80 \
    --use_stemming \
    --epochs 10 \
    --batch_size 32

# Grid search
python src/experiments/automated_grid_search.py \
    --max_iterations 50 \
    --early_stopping_patience 5

# Automated training pipeline
python src/experiments/automated_training.py
```

### Key Findings

- ✅ **Hybrid architecture** outperforms individual models
- ✅ **Stemming preprocessing** improves performance by ~2%
- ✅ **80-20 split** provides optimal train-test balance
- ✅ **GRU hidden size 512** yields best results

### Visualizations

The project generates comprehensive visualizations including:
- Training/validation curves
- Confusion matrices
- Performance comparisons
- Hyperparameter sensitivity analysis

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{sarcasm_detection_indobert_gru,
  title={Sarcasm Detection in Indonesian Text Using IndoBERT and Gated Recurrent Unit},
  author={Your Name},
  journal={Your Journal},
  year={2024},
  url={https://github.com/andiahyaul/Sarcasm-Detection-Using-IndoBERT-and-Gated-Recurrent-Unit-GRU-}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

