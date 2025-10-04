# TimesFM 2.5 Setup Guide

## Installation

### Quick Install (from source)

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .
```

### With All Features

```bash
# Install with covariates support
pip install -e . jax jaxlib scikit-learn

# Or install all dependencies from requirements.txt
pip install -r requirements.txt
```

### In Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt  # For all features
```

## What's Installed

### Core TimesFM
- `torch>=2.0.0` - PyTorch backend
- `numpy>=1.26.4` - Numerical computing
- `huggingface_hub>=0.23.0` - Model downloading
- `safetensors>=0.5.3` - Model format

### Covariates Support (NEW!)
- `jax>=0.4.0`, `jaxlib>=0.4.0` - For linear regression
- `scikit-learn>=1.0.0` - One-hot encoding, preprocessing

### Data Analysis
- `pandas>=2.0.0` - Data manipulation
- `seaborn>=0.13.0` - Visualization
- `matplotlib>=3.8.0` - Plotting

### Jupyter
- `jupyter>=1.0.0` - Notebook environment
- `ipykernel>=6.0.0` - Kernel support
- `ipywidgets>=8.0.0` - Interactive widgets

## Usage

### Basic Forecasting

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
)

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],
)
```

### With Covariates (NEW!)

```python
cov_forecast, xreg_forecast = model.forecast_with_covariates(
    horizon=7,
    inputs=[[30, 30, 4, 5, 7, 8, 10]],  # Historical data
    dynamic_numerical_covariates={
        "temperature": [[31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1,
                        32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]],
    },
    dynamic_categorical_covariates={
        "weekday": [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]],
    },
    static_numerical_covariates={
        "base_price": [1.99],
    },
    static_categorical_covariates={
        "category": ["food"],
    },
    xreg_mode="xreg + timesfm",
)
```

## Examples

- **Basic Usage**: `notebooks/timesfm_example.ipynb`
- **With Covariates**: `notebooks/covariates_2p5_example.ipynb`
- **Documentation**: `README.md`, `COVARIATES_2P5.md`

## Features

### TimesFM 2.5 Highlights
- 200M parameters (down from 500M in v2.0)
- Up to 16k context length (up from 2048)
- Continuous quantile forecasts up to 1k horizon
- No frequency indicator required

### New: Covariates Support
- Static covariates (per time series)
- Dynamic covariates (per time point)
- Numerical and categorical support
- Two XReg modes: "xreg + timesfm" and "timesfm + xreg"

## Files

- `src/timesfm/` - Core TimesFM code
- `src/timesfm/xreg_lib.py` - Covariates support
- `notebooks/` - Example notebooks
- `README.md` - Main documentation
- `COVARIATES_2P5.md` - Covariates documentation
- `requirements.txt` - All dependencies
- `.gitignore` - Git ignore patterns

## Git Ignore

The `.gitignore` includes:
- Python artifacts (`__pycache__`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- Jupyter checkpoints (`.ipynb_checkpoints`)
- Model caches
- IDE configs (`.vscode/`, `.idea/`)

## Resources

- **Paper**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)
- **Hugging Face**: [TimesFM Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)
- **Blog**: [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

