# TimesFM 2.5 Quantile Forecasting for Inventory Planning

## Overview

TimesFM 2.5 is a pretrained time-series foundation model (200M parameters) from Google Research, designed for zero-/few-shot forecasting. A key upgrade in version 2.5 is its optional **quantile forecasting head**, which enables native probabilistic outputs. Instead of only a single point prediction, TimesFM 2.5 can output a distribution of possible future demand values (P10, P20, ..., P90 percentiles).

This capability is especially relevant for inventory planning challenges like VN2, where short-horizon demand forecasts feed into ordering decisions that must balance shortage vs. holding costs. In inventory planning, uncertainty matters as much as accuracy – an under-forecast can lead to stockouts (lost sales), while over-forecasting ties up capital in excess stock.

## Why Quantile Forecasts Improve Inventory Decisions

Traditional inventory planning often relies on a single forecast (mean or median) plus ad-hoc buffers (safety stock). This approach can be myopic because it doesn't explicitly account for the full demand uncertainty. Quantile forecasts, by contrast, directly target different probability levels of demand, which can be aligned to business service goals and cost preferences.

### 1. Service-Level Targeting

In service-oriented supply chains, planners specify a target service level (e.g., "90% chance of no stockout" per replenishment cycle). A quantile forecast naturally maps to this requirement:

- For a 90% cycle service level, use the 90th percentile demand forecast as the order quantity
- A τ=0.90 quantile forecast directly gives the stock level needed so that there is only a 10% risk of running out during lead time
- This builds the desired service level into the forecast itself, avoiding manual z-score computations

**Example from our implementation** (`quantile_inventory_demo.ipynb`):
```python
# P90 policy achieves ~90% service
q90 = quantile_forecast[:, :, 9]  # 90th percentile
service_achieved = np.mean(q90 >= actuals) * 100  # ~90%
```

### 2. Data-Driven Safety Stock Calculation

Safety stock is traditionally an extra buffer derived from forecast error variance (often assuming normal demand). Quantile forecasts offer a more direct and distribution-agnostic way to compute this buffer.

The difference between a high quantile forecast and the median forecast effectively **is** the safety stock needed for that service level:

**Safety Stock = P90 - P50**

For instance, if the model predicts:
- Median (P50) demand: 100 units
- 90th percentile (P90) demand: 130 units
- Safety stock for 90% service: ~30 units

This is more robust than assuming a normal error distribution; the model's learned quantiles inherently capture skewness, intermittency, or fat-tails present in the data. The quantile approach dynamically adjusts the buffer based on context (wider spread for volatile items, narrower for stable items).

### 3. Asymmetric Cost Trade-offs (Newsvendor Logic)

Inventory decisions often face an asymmetry: the cost of under-stocking (stockout) versus over-stocking (holding) are not equal. Quantile forecasts allow you to explicitly account for this asymmetry by choosing a forecast fractile that minimizes expected cost.

**Newsvendor optimal quantile:**
```
Critical Fractile = Cu / (Cu + Co)
```

Where:
- Cu = cost of underage (stockout) per unit
- Co = cost of overage (holding) per unit

**VN2 Example:**
- Shortage cost (Cu): $1.00 per unit
- Holding cost (Co): $0.20 per unit per week
- Critical fractile: 1.0 / (1.0 + 0.2) = **0.8333**
- **Optimal policy: Use P80 quantile**

**From our VN2 submission** (`vn2_submission_quantile.ipynb`):
```python
# Map critical fractile to closest quantile
CRITICAL_RATIO = SHORTAGE_COST / (SHORTAGE_COST + HOLDING_COST)  # 0.8333
quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
closest_idx = np.argmin(np.abs(quantile_levels - CRITICAL_RATIO))
chosen_quantile = quantile_levels[closest_idx]  # P80

# Use P80 forecast as order quantity
optimal_forecast = quantile_forecast[:, :, closest_idx + 1]
```

This approach inherently accounts for the fact that being wrong on the high side is less costly than being wrong on the low side, which a point forecast alone cannot capture.

## Implementation Guide

### Step 1: Enable Quantile Head

**Code example:**
```python
import timesfm
import torch

# Load pretrained model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Compile with quantile head enabled
model.compile(
    timesfm.ForecastConfig(
        max_context=512,
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,  # Enable quantile forecasting
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,  # Ensure monotonic quantiles
    )
)
```

**Key parameters:**
- `use_continuous_quantile_head=True`: Enables probabilistic outputs
- `fix_quantile_crossing=True`: Ensures P10 ≤ P20 ≤ ... ≤ P90 (monotonic)
- `infer_is_positive=True`: Useful for demand forecasting (non-negative values)

### Step 2: Generate Quantile Forecasts

```python
# Prepare input data
inputs = [...]  # List of time series (each is a numpy array)
horizon = 3  # Forecast 3 weeks ahead

# Generate forecasts
point_forecast, quantile_forecast = model.forecast(
    horizon=horizon,
    inputs=inputs,
)

# Output shapes:
# point_forecast: (N_series, horizon)
# quantile_forecast: (N_series, horizon, 10)
#   Index 0: mean
#   Index 1-9: P10, P20, P30, P40, P50, P60, P70, P80, P90
```

### Step 3: Select Policy-Appropriate Quantile

**Service-Level Policy:**
```python
# Target 90% service → use P90
q90 = quantile_forecast[:, :, 9]
order_qty = q90.sum(axis=1)  # Sum over horizon for protection period
```

**Newsvendor Policy:**
```python
# Cost-optimal quantile
critical_fractile = shortage_cost / (shortage_cost + holding_cost)
quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
closest_idx = np.argmin(np.abs(quantile_levels - critical_fractile))

# Extract chosen quantile
optimal_quantile = quantile_forecast[:, :, closest_idx + 1]
order_qty = optimal_quantile.sum(axis=1)
```

### Step 4: Integrate with Base-Stock Policy

**From our VN2 submission:**
```python
from vn2inventory.policy import compute_orders

# Prepare demand statistics
demand_stats = pd.DataFrame({
    'mean_demand': optimal_quantile.sum(axis=1),  # 3-week demand
    'std_demand': (q80 - q50).sum(axis=1) / 0.84,  # Estimate from IQR
})

# Compute orders
orders = compute_orders(
    index_df=sku_list,
    demand_stats=demand_stats,
    current_state=inventory_state,
    lead_time_weeks=2,
    review_period_weeks=1,
    shortage_cost_per_unit=1.0,
    holding_cost_per_unit_per_week=0.2,
)
```

## Validation: Calibration Check

It's important to validate that quantiles match empirical frequencies:

```python
def check_calibration(actuals, quantile_forecasts, quantile_levels):
    """Verify that P90 contains 90% of actuals, etc."""
    results = {}
    for i, q_level in enumerate(quantile_levels):
        q_forecast = quantile_forecasts[:, :, i + 1]
        coverage = np.mean(actuals <= q_forecast)
        results[f"P{int(q_level*100)}"] = {
            "target": q_level,
            "empirical": coverage,
            "error": coverage - q_level
        }
    return results

# Example output:
# P90: target=0.90, empirical=0.89, error=-0.01 ✓ Well calibrated!
```

## Quantile Forecasts vs. Point Forecasts

### Decision-Optimality vs. Accuracy

A point forecast (mean or median) minimizes forecast error (MAE, MAPE) but may not translate to optimal decisions. By definition:
- Mean minimizes squared errors
- Median minimizes absolute errors
- Neither accounts for **asymmetric costs**

Quantile forecasts are explicitly biased in a useful way: they purposefully tilt the forecast up or down to hit the service level or cost target. The result is often higher customer service and more appropriate inventory levels, even if the quantile forecast is less "accurate" on average than the mean forecast.

**In supply chain, the cost of error matters more than the error itself.**

### vs. Traditional Probabilistic Models

Before TimesFM's quantile head, one would use:
- ARIMA with confidence intervals (assumes normality)
- Prophet with uncertainty intervals (parametric)
- Croston's method (for intermittent demand)
- DeepAR (outputs parametric distribution)

TimesFM's quantile output is **non-parametric** and learned from data, which better captures:
- Skewed distributions (common in retail)
- Intermittent demand (many zeros)
- Fat-tailed demand (rare spikes)

### When to Use Point Forecasts

Not every decision requires full distributions. Point forecasts are still essential for:
- Reporting aggregate demand
- Cost symmetry scenarios
- Competition metrics (MAPE/WMAPE)

TimesFM conveniently provides both in one go. One strategy: use the point forecast for baseline planning and quantiles for stress-testing or policy buffers.

## Practical Notebooks

We provide two ready-to-use notebooks demonstrating TimesFM 2.5 quantile forecasting:

### 1. `notebooks/quantile_inventory_demo.ipynb`
**Educational demo showing key concepts:**
- Enable quantile head
- Generate P10-P90 forecasts
- Service-level policies (P90 → 90% service)
- Newsvendor optimization (cost-optimal quantile)
- Calibration validation
- Safety stock calculation

**Best for:** Learning quantile concepts and validation

### 2. `notebooks/vn2_submission_quantile.ipynb`
**Production-ready VN2 submission:**
- All 599 SKUs
- VN2 cost structure (Cu=$1.0, Co=$0.2)
- Critical fractile calculation (0.833 → P80)
- 3-week protection period aggregation
- Base-stock policy integration
- Generates `orders_timesfm_quantile_demo.csv`

**Best for:** End-to-end submission pipeline

## Key Takeaways

1. **Service-Level Targeting**: Use P90 for ~90% service (no arbitrary buffers)
2. **Cost Optimization**: Choose quantile by Cu/(Cu+Co) ratio
3. **Data-Driven Safety Stock**: Quantile spread captures true uncertainty
4. **Non-Parametric**: No distribution assumptions needed
5. **Zero-Shot**: No per-SKU training required
6. **Calibrated**: Validate that quantiles match empirical frequencies

## Recommended Workflow

1. Enable `use_continuous_quantile_head=True`
2. Generate quantile forecasts for your SKUs
3. Define policy:
   - Service target → use corresponding quantile (P90 for 90%)
   - Cost ratio → compute critical fractile and use newsvendor quantile
4. Validate calibration on held-out data
5. Simulate inventory outcomes (stockouts, holding) before deploying
6. Monitor and adjust quantile choice based on realized costs

## Why This Matters

Quantile forecasts transform predictive uncertainty from a source of risk into a managed input for smarter inventory strategies. Instead of "forecast then add safety stock," you **forecast at the required service/cost level**.

This ensures inventory levels are neither too lean nor too bloated, but rather aligned to your service goals and risk appetite under real-world demand variability.

**"Inventory planning rewards clarity more than cleverness."**

Quantile forecasts contribute to that clarity by explicitly showing the range of demand outcomes and aligning decisions (service levels, safety stocks) with quantifiable probabilities.

## References

- Google Research (2024), *TimesFM: A decoder-only foundation model for time-series forecasting*
- Google TimesFM GitHub, Release notes for 2.5 (quantile head support)
- Our implementation: `notebooks/quantile_inventory_demo.ipynb` and `notebooks/vn2_submission_quantile.ipynb`
- VN2 Inventory Planning Challenge data and benchmark
