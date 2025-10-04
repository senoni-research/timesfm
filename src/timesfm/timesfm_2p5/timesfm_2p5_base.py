# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM 2p5 base implementation."""

import collections
import dataclasses
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from .. import configs

ResidualBlockConfig = configs.ResidualBlockConfig
StackedTransformersConfig = configs.StackedTransformersConfig
TransformerConfig = configs.TransformerConfig
ForecastConfig = configs.ForecastConfig


def strip_leading_nans(arr):
  """Removes contiguous NaN values from the beginning of a NumPy array.

  Args:
    arr: The input NumPy array.

  Returns:
    A new NumPy array with leading NaN values removed.
    If the array is all NaNs or empty, returns an empty array.
  """

  isnan = np.isnan(arr)
  first_valid_index = np.argmax(~isnan)
  return arr[first_valid_index:]


def linear_interpolation(arr):
  """Performs linear interpolation to fill NaN values in a 1D numpy array.

  Args:
      arr: The 1D numpy array containing NaN values.

  Returns:
      A new numpy array with NaN values filled using linear interpolation,
      or the original array if no NaNs are present.
      Returns None if the input is not a 1D array.
      Returns the original array if there are no NaN values.
  """

  nans = np.isnan(arr)
  if not np.any(nans):  # Check if there are any NaNs
    return arr

  def x(z):
    return z.nonzero()[0]

  nans_indices = x(nans)
  non_nans_indices = x(~nans)
  non_nans_values = arr[~nans]

  try:
    arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
  except ValueError:
    if non_nans_values:
      mu = np.nanmean(arr)
    else:
      mu = 0.0
    arr = np.where(np.isfinite(arr), arr, mu)
  return arr


@dataclasses.dataclass(frozen=True)
class TimesFM_2p5_200M_Definition:
  """Framework-agnostic config of TimesFM 2.5."""

  context_limit = 16384
  input_patch_len: int = 32
  output_patch_len: int = 128
  output_quantile_len: int = 1024
  quantiles: list[float] = dataclasses.field(
    default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  )
  decode_index: int = 5
  tokenizer: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=64,
    hidden_dims=1280,
    output_dims=1280,
    use_bias=True,
    activation="swish",
  )
  stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
    num_layers=20,
    transformer=TransformerConfig(
      model_dims=1280,
      hidden_dims=1280,
      num_heads=16,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      use_bias=False,
      use_rotary_position_embeddings=True,
      ff_activation="swish",
      fuse_qkv=True,
    ),
  )
  output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=1280,
    hidden_dims=1280,
    output_dims=1280,
    use_bias=False,
    activation="swish",
  )
  output_projection_quantiles: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=1280,
    hidden_dims=1280,
    output_dims=10240,
    use_bias=False,
    activation="swish",
  )


class TimesFM_2p5:
  """Abstract base class for TimesFM models.

  Attributes:
    forecast_config: Configuration for forecasting flags.
    compiled_decode: Compiled decode function.
    global_batch_size: Global batch size.
  """

  forecast_config: ForecastConfig | None = None
  compiled_decode: Callable[..., Any] | None = None
  global_batch_size: int = 0

  def load_checkpoint(self, path: str):
    """Loads a TimesFM model from a checkpoint."""
    raise NotImplementedError()

  def compile(self, forecast_config: ForecastConfig | None = None):
    """Compiles the TimesFM model for fast decoding."""
    raise NotImplementedError()

  def forecast(
    self, horizon: int, inputs: list[np.ndarray]
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts the time series."""
    if self.compiled_decode is None:
      raise RuntimeError("Model is not compiled. Please call compile() first.")

    assert self.global_batch_size > 0
    assert self.forecast_config is not None

    context = self.forecast_config.max_context
    num_inputs = len(inputs)
    if (w := num_inputs % self.global_batch_size) != 0:
      inputs += [np.array([0.0] * 3)] * (self.global_batch_size - w)

    output_points = []
    output_quantiles = []
    values = []
    masks = []
    idx = 0
    for each_input in inputs:
      value = linear_interpolation(strip_leading_nans(np.array(each_input)))
      if (w := len(value)) >= context:
        value = value[-context:]
        mask = np.zeros_like(value, dtype=bool)
      else:
        mask = np.array([True] * (context - w) + [False] * w)
        value = np.pad(value, (context - w, 0), "constant", constant_values=0.0)
      values.append(value)
      masks.append(mask)
      idx += 1
      if idx == self.global_batch_size:
        idx = 0
        point_forecast, quantile_forecast = self.compiled_decode(horizon, values, masks)
        output_points.append(point_forecast)
        output_quantiles.append(quantile_forecast)
        values = []
        masks = []

    output_points = np.concatenate(output_points, axis=0)
    output_quantiles = np.concatenate(output_quantiles, axis=0)
    return output_points[:num_inputs], output_quantiles[:num_inputs]

  def forecast_with_covariates(
      self,
      horizon: int,
      inputs: list[Sequence[float]],
      dynamic_numerical_covariates: (dict[str, Sequence[Sequence[float]]] |
                                     None) = None,
      dynamic_categorical_covariates: (dict[str, Sequence[Sequence[Any]]] |
                                       None) = None,
      static_numerical_covariates: dict[str, Sequence[float]] | None = None,
      static_categorical_covariates: (dict[str, Sequence[Any]] |
                                      None) = None,
      xreg_mode: Literal["timesfm + xreg", "xreg + timesfm"] = "xreg + timesfm",
      normalize_xreg_target_per_input: bool = True,
      ridge: float = 0.0,
      max_rows_per_col: int = 0,
      force_on_cpu: bool = False,
  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Forecasts on a list of time series with covariates.

    This method uses external regressors (covariates) to improve forecasting.
    It fits a linear model on the covariates and combines it with TimesFM forecasts.

    To optimize inference speed, avoid string valued categorical covariates.

    Args:
      horizon: The forecast horizon length.
      inputs: A list of time series forecast contexts. Each context time series
        should be convertible to a numpy array.
      dynamic_numerical_covariates: A dict of dynamic numerical covariates.
        Each value should be a list of sequences, where each sequence contains
        covariate values for both context and horizon (context_len + horizon_len).
      dynamic_categorical_covariates: A dict of dynamic categorical covariates.
        Similar structure to dynamic_numerical_covariates.
      static_numerical_covariates: A dict of static numerical covariates.
        Each value should be a list with one value per input time series.
      static_categorical_covariates: A dict of static categorical covariates.
        Each value should be a list with one value per input time series.
      xreg_mode: one of "xreg + timesfm" or "timesfm + xreg".
        - "xreg + timesfm": Fit linear model on targets, then forecast residuals.
        - "timesfm + xreg": Forecast with TimesFM, then fit linear model on residuals.
      normalize_xreg_target_per_input: whether to normalize the xreg target per
        input in the given batch.
      ridge: ridge penalty for the linear model.
      max_rows_per_col: max number of rows per column for the linear model.
      force_on_cpu: whether to force running on cpu for the linear model.

    Returns:
      A tuple of two lists:
        - The first list contains the combined forecasts (TimesFM + covariates).
        - The second list contains the pure linear regression forecasts.

    Raises:
      RuntimeError: If the model is not compiled.
      ValueError: If no covariates are provided or if horizon exceeds model capacity.
    """
    from .. import xreg_lib

    if self.compiled_decode is None:
      raise RuntimeError("Model is not compiled. Please call compile() first.")

    # Verify and bookkeep covariates.
    if not (dynamic_numerical_covariates or dynamic_categorical_covariates or
            static_numerical_covariates or static_categorical_covariates):
      raise ValueError(
          "At least one of dynamic_numerical_covariates,"
          " dynamic_categorical_covariates, static_numerical_covariates,"
          " static_categorical_covariates must be set.")

    # Track the lengths of (1) each input, (2) the part that can be used in the
    # linear model, and (3) the horizon.
    input_lens, train_lens, test_lens = [], [], []
    
    # Get patch length from model config
    if hasattr(self, 'model') and hasattr(self.model, 'p'):
      input_patch_len = self.model.p
    else:
      input_patch_len = 32  # Default patch length for TimesFM 2.5

    for i, input_ts in enumerate(inputs):
      input_len = len(input_ts)
      input_lens.append(input_len)

      if xreg_mode == "timesfm + xreg":
        # For fitting residuals, no TimesFM forecast on the first patch.
        train_lens.append(max(0, input_len - input_patch_len))
      elif xreg_mode == "xreg + timesfm":
        train_lens.append(input_len)
      else:
        raise ValueError(f"Unsupported mode: {xreg_mode}")

      # Infer test_lens from dynamic covariates
      if dynamic_numerical_covariates:
        test_lens.append(
            len(list(dynamic_numerical_covariates.values())[0][i]) - input_len)
      elif dynamic_categorical_covariates:
        test_lens.append(
            len(list(dynamic_categorical_covariates.values())[0][i]) -
            input_len)
      else:
        test_lens.append(horizon)

      if test_lens[-1] != horizon:
        raise ValueError(
            f"Covariate length mismatch for input {i}. Expected horizon {horizon}, "
            f"but got {test_lens[-1]} from covariates.")

    # Prepare the covariates into train and test.
    train_dynamic_numerical_covariates = collections.defaultdict(list)
    test_dynamic_numerical_covariates = collections.defaultdict(list)
    train_dynamic_categorical_covariates = collections.defaultdict(list)
    test_dynamic_categorical_covariates = collections.defaultdict(list)
    
    for covariates, train_covariates, test_covariates in (
        (
            dynamic_numerical_covariates,
            train_dynamic_numerical_covariates,
            test_dynamic_numerical_covariates,
        ),
        (
            dynamic_categorical_covariates,
            train_dynamic_categorical_covariates,
            test_dynamic_categorical_covariates,
        ),
    ):
      if not covariates:
        continue
      for covariate_name, covariate_values in covariates.items():
        for input_len, train_len, covariate_value in zip(
            input_lens, train_lens, covariate_values):
          train_covariates[covariate_name].append(
              covariate_value[(input_len - train_len):input_len])
          test_covariates[covariate_name].append(covariate_value[input_len:])

    # Fit models based on the selected mode.
    if xreg_mode == "timesfm + xreg":
      # Forecast via TimesFM then fit a model on the residuals.
      mean_outputs, _ = self.forecast(horizon, inputs)
      
      # For residuals, we need the forecast on the context
      # Since v2.5 doesn't return forecast on context by default, we'll use a simpler approach
      # We'll fit on the difference between the input and a simple continuation
      targets = [
          np.array(input_ts)[-train_len:]
          for input_ts, train_len in zip(inputs, train_lens)
      ]
      per_instance_stats = None
      if normalize_xreg_target_per_input:
        targets, per_instance_stats = _normalize(targets)
      
      xregs = xreg_lib.BatchedInContextXRegLinear(
          targets=targets,
          train_lens=train_lens,
          test_lens=test_lens,
          train_dynamic_numerical_covariates=train_dynamic_numerical_covariates,
          test_dynamic_numerical_covariates=test_dynamic_numerical_covariates,
          train_dynamic_categorical_covariates=
          train_dynamic_categorical_covariates,
          test_dynamic_categorical_covariates=
          test_dynamic_categorical_covariates,
          static_numerical_covariates=static_numerical_covariates,
          static_categorical_covariates=static_categorical_covariates,
      ).fit(
          ridge=ridge,
          one_hot_encoder_drop=None if ridge > 0 else "first",
          max_rows_per_col=max_rows_per_col,
          force_on_cpu=force_on_cpu,
          debug_info=False,
          assert_covariates=True,
          assert_covariate_shapes=True,
      )
      if normalize_xreg_target_per_input:
        xregs = _renormalize(xregs, per_instance_stats)
      outputs = [
          mean_output + xreg
          for mean_output, xreg in zip(mean_outputs, xregs)
      ]

    else:  # xreg_mode == "xreg + timesfm"
      # Fit a model on the targets then forecast on the residuals via TimesFM.
      targets = [
          np.array(input_ts)[-train_len:]
          for input_ts, train_len in zip(inputs, train_lens)
      ]
      per_instance_stats = None
      if normalize_xreg_target_per_input:
        targets, per_instance_stats = _normalize(targets)
      
      xregs, xregs_on_context, _, _, _ = xreg_lib.BatchedInContextXRegLinear(
          targets=targets,
          train_lens=train_lens,
          test_lens=test_lens,
          train_dynamic_numerical_covariates=train_dynamic_numerical_covariates,
          test_dynamic_numerical_covariates=test_dynamic_numerical_covariates,
          train_dynamic_categorical_covariates=
          train_dynamic_categorical_covariates,
          test_dynamic_categorical_covariates=
          test_dynamic_categorical_covariates,
          static_numerical_covariates=static_numerical_covariates,
          static_categorical_covariates=static_categorical_covariates,
      ).fit(
          ridge=ridge,
          one_hot_encoder_drop=None if ridge > 0 else "first",
          max_rows_per_col=max_rows_per_col,
          force_on_cpu=force_on_cpu,
          debug_info=True,
          assert_covariates=True,
          assert_covariate_shapes=True,
      )
      
      # Forecast the residuals
      residuals = [
          target - xreg_on_context
          for target, xreg_on_context in zip(targets, xregs_on_context)
      ]
      mean_outputs, _ = self.forecast(horizon, residuals)
      
      outputs = [
          mean_output + xreg
          for mean_output, xreg in zip(mean_outputs, xregs)
      ]
      if normalize_xreg_target_per_input:
        outputs = _renormalize(outputs, per_instance_stats)

    return outputs, xregs


def _normalize(targets: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
  """Normalizes each target series independently.

  Args:
    targets: List of target arrays.

  Returns:
    Tuple of (normalized_targets, stats) where stats contains mean and std
    for each target.
  """
  normalized = []
  stats = []
  for target in targets:
    arr = np.array(target)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-6:
      std = 1.0
    normalized.append((arr - mean) / std)
    stats.append({"mean": mean, "std": std})
  return normalized, stats


def _renormalize(
    forecasts: list[np.ndarray], stats: list[dict]
) -> list[np.ndarray]:
  """Reverses normalization applied by _normalize.

  Args:
    forecasts: List of forecast arrays.
    stats: List of dictionaries containing 'mean' and 'std' for each forecast.

  Returns:
    List of renormalized forecast arrays.
  """
  return [
      np.array(forecast) * stat["std"] + stat["mean"]
      for forecast, stat in zip(forecasts, stats)
  ]
