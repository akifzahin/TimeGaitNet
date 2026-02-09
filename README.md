# TimeGaitNet: BiMamba-based Multitask Learning for Freezing of Gait Detection

TimeGaitNet is a deep learning framework for detecting Freezing of Gait (FOG) episodes in Parkinson's disease patients using wearable accelerometer data. The system combines bidirectional Mamba (BiMamba) state space models with multiscale CNNs for multitask learning across FOG detection, intensity estimation, and forecasting.

## Features

- **Multitask Learning**: Simultaneous FOG detection, intensity estimation, and 5-second forecasting
- **Frequency-Aware Gating**: Novel mechanism to capture clinically-relevant FOG frequencies (3-8 Hz)
- **State Space Models**: Efficient BiMamba encoders for temporal modeling
- **Robust Evaluation**: Multi-seed experiments with comprehensive metrics



## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TimeGaitNet.git
cd TimeGaitNet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

4. Install Mamba SSM (required for BiMamba models):
```bash
pip install mamba-ssm causal-conv1d
```

## Quick Start

### Running Experiments

1. Update dataset paths in `experiments/run_ablation.py`:
```python
tdcsfog_train_path = '/path/to/tdcsfog_train.parquet'
tdcsfog_test_path = '/path/to/tdcsfog_test.parquet'
daphnet_train_path = '/path/to/daphnet_train.parquet'
daphnet_test_path = '/path/to/daphnet_test.parquet'
```

2. Run ablation study:
```bash
python experiments/run_ablation.py
```

This will:
- Train BiMamba and BiMamba-FreqAware models across 5 seeds
- Evaluate on DAPHNET and TDCSFOG datasets
- Generate comprehensive plots and metrics
- Save results to CSV files

### Using Individual Components

```python
from config.model_configs import create_bimamba_freq_aware
from models.base_model import MultitaskFOGModel
from data.dataset import FOGDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = FOGDataset('path/to/data.parquet', augment=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create model
config = create_bimamba_freq_aware()
model = MultitaskFOGModel(config)

# Training loop
for batch in loader:
    X = batch['X']
    predictions = model(X)
    # predictions contains 'binary', 'intensity', 'forecast'
```

## Model Architectures

### BiMamba (Standard)
- Multiscale CNN feature extraction
- Bidirectional Mamba temporal encoder
- Gated fusion of spatial and temporal features
- Separate task-specific heads

### BiMamba-FreqAware
- All BiMamba components
- Frequency extractor (3-8 Hz FOG band)
- Frequency-aware gating mechanism
- Enhanced capture of clinically-relevant patterns

## Results

### Performance (5-seed average on test sets)

**DAPHNET Dataset:**
- Binary F1: 88.5% ± 1.2%
- Binary PR-AUC: 91.3% ± 0.8%
- Intensity MSE: 0.024 ± 0.003
- Forecast F1: 85.2% ± 1.5%

**TDCSFOG Dataset:**
- Binary F1: 95.1% ± 0.6%
- Binary PR-AUC: 97.2% ± 0.4%
- Intensity MSE: 0.018 ± 0.002
- Forecast F1: 92.8% ± 0.9%

### Model Efficiency
- Parameters: 4.21M
- Inference time: ~12ms per window (GPU)
- Memory footprint: ~850MB

## Configuration

Model configurations can be customized in `config/model_configs.py`:

```python
config = ModelConfig()
config.temporal_model_type = 'bimamba'
config.use_frequency_gating = True  # Enable frequency-aware variant
config.sharing_mode = 'no_sharing'  # Task-specific encoders
config.dropout = 0.3
config.binary_weight = 0.33  # Task weighting
```

## Training Details

- **Optimizer**: AdamW (lr=5e-4, weight_decay=5e-4)
- **Scheduler**: 5-epoch warmup + cosine annealing
- **Batch size**: 64
- **Epochs**: 30 (with early stopping, patience=10)
- **Loss**: Adaptive multitask loss with dynamic weighting
- **Data augmentation**: Noise injection, time warping, magnitude scaling

## Evaluation Metrics

- **Binary Detection**: F1, Precision, Recall, PR-AUC, ROC-AUC
- **Intensity Estimation**: MSE, MAE
- **Forecasting**: F1, PR-AUC, ROC-AUC
- **Visualization**: ROC curves, PR curves, confusion matrices, residual plots


## License

This project is licensed under the MIT License - see the LICENSE file for details.

