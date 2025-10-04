# Covariates Support for TimesFM 2.5

This document describes the covariates (external regressors) support implemented for TimesFM 2.5, based on the v1 implementation.

## Overview

Covariates support allows you to incorporate external information (e.g., weather, promotions, calendar features) into TimesFM forecasts to improve accuracy. The implementation uses a batched in-context linear regression approach combined with TimesFM's deep learning forecasts.

## Implementation Details

### Files Modified/Added

1. **`src/timesfm/xreg_lib.py`** (copied from v1)
   - `BatchedInContextXRegBase`: Base class for covariate handling
   - `BatchedInContextXRegLinear`: Linear regression implementation using JAX
   - Supports normalization, one-hot encoding, and ridge regression

2. **`src/timesfm/timesfm_2p5/timesfm_2p5_base.py`** (modified)
   - Added `forecast_with_covariates()` method to `TimesFM_2p5` class
   - Added helper functions `_normalize()` and `_renormalize()`
   - Imports required for covariate handling

3. **`pyproject.toml`** (modified)
   - Added optional dependencies: `jax`, `jaxlib`, `scikit-learn`
   - Created `[covariates]` extras group

4. **`README.md`** (modified)
   - Added covariates section with examples
   - Marked covariate support as completed

5. **`notebooks/covariates_2p5_example.ipynb`** (new)
   - Complete working example with grocery store sales forecasting
   - Demonstrates all covariate types and modes

## API Usage

### Method Signature

```python
def forecast_with_covariates(
    self,
    horizon: int,
    inputs: list[Sequence[float]],
    dynamic_numerical_covariates: dict[str, Sequence[Sequence[float]]] | None = None,
    dynamic_categorical_covariates: dict[str, Sequence[Sequence[Any]]] | None = None,
    static_numerical_covariates: dict[str, Sequence[float]] | None = None,
    static_categorical_covariates: dict[str, Sequence[Any]] | None = None,
    xreg_mode: Literal["timesfm + xreg", "xreg + timesfm"] = "xreg + timesfm",
    normalize_xreg_target_per_input: bool = True,
    ridge: float = 0.0,
    max_rows_per_col: int = 0,
    force_on_cpu: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
```

### Covariate Types

1. **Static Numerical**: One numerical value per time series
   - Example: `{"base_price": [10.5, 15.2]}`

2. **Static Categorical**: One categorical value per time series
   - Example: `{"category": ["food", "beverage"]}`

3. **Dynamic Numerical**: Numerical values for each time point (context + horizon)
   - Example: `{"temperature": [[20.1, 21.3, ...], [18.5, 19.2, ...]]}`

4. **Dynamic Categorical**: Categorical values for each time point (context + horizon)
   - Example: `{"weekday": [[0, 1, 2, ...], [3, 4, 5, ...]]}`

### XReg Modes

1. **"xreg + timesfm"** (recommended):
   - Fit linear model on the targets
   - Forecast the residuals with TimesFM
   - Combine both forecasts
   - Better for strong linear relationships

2. **"timesfm + xreg"**:
   - Forecast with TimesFM first
   - Fit linear model on the residuals
   - Combine both forecasts
   - Better when TimesFM captures most patterns

## Installation

### Basic Installation

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .
```

### With Covariates Support

```bash
pip install -e ".[covariates]"
```

Or install dependencies separately:

```bash
pip install jax jaxlib scikit-learn
```

## Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

# Load and compile model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=512,
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
)

# Prepare data
inputs = [
    [30.0, 30.0, 4.0, 5.0, 7.0, 8.0, 10.0],  # Ice cream sales
    [5.0, 7.0, 12.0, 13.0, 5.0, 6.0, 10.0],  # Sunscreen sales
]

# Forecast with covariates
cov_forecast, xreg_forecast = model.forecast_with_covariates(
    horizon=7,
    inputs=inputs,
    dynamic_numerical_covariates={
        "temperature": [
            [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 
             32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2],
            [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 
             32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2],
        ],
    },
    dynamic_categorical_covariates={
        "weekday": [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
        ],
    },
    static_numerical_covariates={
        "base_price": [1.99, 29.99],
    },
    static_categorical_covariates={
        "category": ["food", "skin_product"],
    },
    xreg_mode="xreg + timesfm",
)

print(f"Combined forecast: {cov_forecast}")
print(f"Linear model forecast: {xreg_forecast}")
```

## Technical Details

### Linear Regression Implementation

- Uses JAX for efficient matrix operations
- Supports ridge regression for regularization
- One-hot encoding for categorical variables
- Automatic normalization of numerical features
- Option to run on CPU or GPU

### Differences from v1

1. **API Changes**:
   - v2.5 uses `horizon` parameter instead of inferring from covariates
   - Simplified return values (no context forecasts by default)

2. **Model Architecture**:
   - v2.5 has different patch sizes (32 vs context-dependent in v1)
   - Different internal representation of forecasts

3. **Dependencies**:
   - v2.5 requires Python >= 3.11 (vs 3.10 for v1)
   - Uses newer versions of JAX/PyTorch

## References

- Original v1 implementation: `v1/src/timesfm/timesfm_base.py`
- v1 covariates notebook: `v1/notebooks/covariates.ipynb`
- v1 README covariates section: `v1/README.md`

## Future Enhancements

Potential improvements:
1. Add support for past-only dynamic covariates
2. Optimize linear regression for large covariate matrices
3. Add more sophisticated regression methods (e.g., lasso, elastic net)
4. Support for hierarchical/grouped time series with shared covariates

