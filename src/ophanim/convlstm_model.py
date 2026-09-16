"""Compact causal ConvLSTM baseline for native CODE VTEC grids.

The model predicts the *next* native-grid map from the preceding twelve
two-hour maps.  Its output is a normalized residual relative to the same UTC
slot one day earlier (the first frame in the default history window).  The
target map is never an input to :class:`PredictiveConvLSTM` or
:func:`predict_next`.

PyTorch is deliberately optional.  Importing this module and inspecting its
versioned configuration contract works without PyTorch; tensor preparation,
training, inference, and artifact restoration raise
:class:`ConvLSTMUnavailableError` with an installation hint when it is absent.

Artifacts and training checkpoints use canonical JSON plus base64-encoded
CPU tensors instead of pickle.  This makes their bytes deterministic, keeps
the accepted schema narrow, and gives resumable jobs both model and AdamW
optimizer state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, ClassVar, Self

try:  # The core application remains usable without the optional ML stack.
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _functional
except ImportError:  # pragma: no cover - the default test environment path.
    _torch = None
    _nn = None
    _functional = None


ALGORITHM = "ophanim-convlstm-prior-day-residual/1"
ARTIFACT_FORMAT = "ophanim-convlstm-model/1"
CHECKPOINT_FORMAT = "ophanim-convlstm-checkpoint/1"
NATIVE_GRID_HEIGHT = 71
NATIVE_GRID_WIDTH = 72
TORCH_AVAILABLE = _torch is not None

_MODEL_INPUT_CHANNELS = 2  # normalized VTEC and an explicit validity mask
_SECONDS_PER_DAY = 86_400
_MAX_SERIALIZED_BYTES = 128 * 1024 * 1024
_SUPPORTED_TENSOR_DTYPES = {
    "bool": "bool",
    "float32": "float32",
    "float64": "float64",
    "int64": "int64",
}


class ConvLSTMError(ValueError):
    """A ConvLSTM configuration, tensor, or operation is invalid."""


class ConvLSTMUnavailableError(RuntimeError):
    """The optional PyTorch runtime is required for this operation."""


class ConvLSTMArtifactError(ConvLSTMError):
    """Serialized model or checkpoint bytes violate their contract."""


@dataclass(frozen=True, slots=True)
class ConvLSTMConfig:
    """Versioned architecture and optimization configuration.

    Defaults match the 71 by 72 native CODE grid and twelve two-hour history
    frames.  ``reference_lag_frames * cadence_seconds`` must be exactly one
    day so the residual baseline always represents the same UTC slot.
    """

    algorithm: ClassVar[str] = ALGORITHM

    grid_height: int = NATIVE_GRID_HEIGHT
    grid_width: int = NATIVE_GRID_WIDTH
    history_frames: int = 12
    reference_lag_frames: int = 12
    cadence_seconds: int = 7_200
    hidden_channels: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    initialization_seed: int = 17_291

    def __post_init__(self) -> None:
        for name in (
            "grid_height",
            "grid_width",
            "history_frames",
            "reference_lag_frames",
            "cadence_seconds",
            "hidden_channels",
            "kernel_size",
            "initialization_seed",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConvLSTMError(f"{name} must be an integer")
        if self.grid_height < 2 or self.grid_width < 2:
            raise ConvLSTMError("grid dimensions must both be at least 2")
        if not 1 <= self.history_frames <= 96:
            raise ConvLSTMError("history_frames must be in [1, 96]")
        if not 1 <= self.reference_lag_frames <= self.history_frames:
            raise ConvLSTMError(
                "reference_lag_frames must be in [1, history_frames]"
            )
        if self.cadence_seconds <= 0:
            raise ConvLSTMError("cadence_seconds must be positive")
        if self.reference_lag_frames * self.cadence_seconds != _SECONDS_PER_DAY:
            raise ConvLSTMError(
                "reference lag and cadence must describe exactly one UTC day"
            )
        if not 1 <= self.hidden_channels <= 128:
            raise ConvLSTMError("hidden_channels must be in [1, 128]")
        if (
            self.kernel_size < 1
            or self.kernel_size > 9
            or self.kernel_size % 2 == 0
        ):
            raise ConvLSTMError("kernel_size must be an odd integer in [1, 9]")
        padding = self.kernel_size // 2
        if padding > self.grid_width:
            raise ConvLSTMError("kernel padding exceeds the longitude width")
        if not 0 <= self.initialization_seed < 2**63:
            raise ConvLSTMError("initialization_seed must be in [0, 2**63)")
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConvLSTMError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ConvLSTMError(f"{name} must be a finite number")
        if self.learning_rate <= 0.0:
            raise ConvLSTMError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ConvLSTMError("weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ConvLSTMError("gradient_clip_norm must be positive")

    @property
    def reference_history_index(self) -> int:
        """Index of the prior-day reference inside a history window."""

        return self.history_frames - self.reference_lag_frames

    @property
    def configuration_sha256(self) -> str:
        """Stable identity for architecture and training behavior."""

        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, int | float]:
        """Return the exact JSON-safe configuration contract."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    def to_bytes(self) -> bytes:
        """Serialize the configuration as canonical finite JSON."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        """Restore an exact canonical configuration, rejecting extensions."""

        payload = _parse_canonical_json(content, maximum_bytes=64 * 1024)
        if not isinstance(payload, dict):
            raise ConvLSTMArtifactError("configuration root must be an object")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ConvLSTMArtifactError(
                "configuration fields do not match the supported contract"
            )
        try:
            return cls(**payload)
        except (TypeError, ValueError) as error:
            if isinstance(error, ConvLSTMError):
                raise
            raise ConvLSTMArtifactError("configuration values are invalid") from error


@dataclass(frozen=True, slots=True)
class GridNormalization:
    """Robust scalar normalization fitted only on the training period."""

    center_tecu: float
    scale_tecu: float

    def __post_init__(self) -> None:
        for name, value in (
            ("center_tecu", self.center_tecu),
            ("scale_tecu", self.scale_tecu),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ConvLSTMError(f"{name} must be a finite number")
        if self.scale_tecu <= 0.0:
            raise ConvLSTMError("scale_tecu must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "center_tecu": float(self.center_tecu),
            "scale_tecu": float(self.scale_tecu),
        }


@dataclass(frozen=True, slots=True)
class CausalGridBatch:
    """Tensor batch with model inputs kept separate from supervised targets.

    Shapes are ``inputs[B,T,2,H,W]`` and ``reference_vtec``,
    ``target_residual``, ``target_vtec``, and ``loss_mask`` all
    ``[B,1,H,W]``.  Only ``inputs`` are accepted by the model's ``forward``.
    """

    inputs: Any
    reference_vtec: Any
    target_residual: Any
    target_vtec: Any
    loss_mask: Any
    target_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GridPrediction:
    """CPU prediction tensors for the next native-grid readout."""

    predicted_vtec_tecu: Any
    predicted_residual_tecu: Any
    valid_mask: Any


@dataclass(frozen=True, slots=True)
class LoadedConvLSTMModel:
    """Restored inference artifact and its provenance metadata."""

    model: Any
    config: ConvLSTMConfig
    normalization: GridNormalization
    training_data_sha256: str
    completed_epochs: int
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    """Model, optimizer, and cursor restored from a canonical checkpoint."""

    model: Any
    optimizer: Any
    config: ConvLSTMConfig
    normalization: GridNormalization
    training_data_sha256: str
    completed_epochs: int
    next_window_index: int
    checkpoint_sha256: str


if TORCH_AVAILABLE:

    class CircularLongitudeConv2d(_nn.Module):
        """2-D convolution that wraps longitude but zero-pads latitude."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            *,
            bias: bool = True,
        ) -> None:
            super().__init__()
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ConvLSTMError("convolution kernel size must be positive and odd")
            self.padding = kernel_size // 2
            self.convolution = _nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                bias=bias,
            )

        def forward(self, values: Any) -> Any:
            padding = self.padding
            if padding:
                # F.pad orders 2-D padding as left, right, top, bottom.  Apply
                # wrapping only along longitude, then independent zeros at the
                # polar/latitude boundary.
                values = _functional.pad(
                    values,
                    (padding, padding, 0, 0),
                    mode="circular",
                )
                values = _functional.pad(
                    values,
                    (0, 0, padding, padding),
                    mode="constant",
                    value=0.0,
                )
            return self.convolution(values)


    class _ConvLSTMCell(_nn.Module):
        def __init__(self, channels: int, hidden_channels: int, kernel_size: int):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.gates = CircularLongitudeConv2d(
                channels + hidden_channels,
                4 * hidden_channels,
                kernel_size,
            )

        def forward(self, values: Any, hidden: Any, memory: Any) -> tuple[Any, Any]:
            gates = self.gates(_torch.cat((values, hidden), dim=1))
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
            input_gate = _torch.sigmoid(input_gate)
            forget_gate = _torch.sigmoid(forget_gate)
            candidate = _torch.tanh(candidate)
            output_gate = _torch.sigmoid(output_gate)
            memory = forget_gate * memory + input_gate * candidate
            hidden = output_gate * _torch.tanh(memory)
            return hidden, memory


    class PredictiveConvLSTM(_nn.Module):
        """Small single-layer next-map ConvLSTM residual predictor."""

        algorithm: ClassVar[str] = ALGORITHM
        artifact_format: ClassVar[str] = ARTIFACT_FORMAT

        def __init__(self, config: ConvLSTMConfig):
            super().__init__()
            self.config = config
            self.encoder = CircularLongitudeConv2d(
                _MODEL_INPUT_CHANNELS,
                config.hidden_channels,
                config.kernel_size,
            )
            self.cell = _ConvLSTMCell(
                config.hidden_channels,
                config.hidden_channels,
                config.kernel_size,
            )
            self.decoder_hidden = CircularLongitudeConv2d(
                config.hidden_channels,
                config.hidden_channels,
                config.kernel_size,
            )
            self.decoder_output = CircularLongitudeConv2d(
                config.hidden_channels,
                1,
                config.kernel_size,
            )

        def forward(self, inputs: Any) -> Any:
            if not _torch.is_tensor(inputs) or inputs.ndim != 5:
                raise ConvLSTMError("model inputs must have shape [B,T,2,H,W]")
            batch, steps, channels, height, width = inputs.shape
            expected = (
                self.config.history_frames,
                _MODEL_INPUT_CHANNELS,
                self.config.grid_height,
                self.config.grid_width,
            )
            if (steps, channels, height, width) != expected:
                raise ConvLSTMError(
                    "model input shape does not match the versioned configuration"
                )
            if not _torch.isfinite(inputs).all().item():
                raise ConvLSTMError("model inputs contain non-finite values")
            hidden = inputs.new_zeros(
                (batch, self.config.hidden_channels, height, width)
            )
            memory = inputs.new_zeros(
                (batch, self.config.hidden_channels, height, width)
            )
            for index in range(steps):
                encoded = _torch.tanh(self.encoder(inputs[:, index]))
                hidden, memory = self.cell(encoded, hidden, memory)
            decoded = _torch.tanh(self.decoder_hidden(hidden))
            return self.decoder_output(decoded)


else:

    class CircularLongitudeConv2d:  # pragma: no cover - trivial unavailable shim.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


    class PredictiveConvLSTM:  # pragma: no cover - trivial unavailable shim.
        algorithm: ClassVar[str] = ALGORITHM
        artifact_format: ClassVar[str] = ARTIFACT_FORMAT

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


def fit_grid_normalization(
    vtec_tecu: Any,
    valid_mask: Any,
    *,
    scale_floor_tecu: float = 0.1,
) -> GridNormalization:
    """Fit median/MAD normalization from training-period valid cells only."""

    _require_torch()
    if not math.isfinite(float(scale_floor_tecu)) or scale_floor_tecu <= 0.0:
        raise ConvLSTMError("scale_floor_tecu must be positive and finite")
    values = _torch.as_tensor(vtec_tecu, dtype=_torch.float32, device="cpu")
    mask = _torch.as_tensor(valid_mask, dtype=_torch.bool, device="cpu")
    if values.shape != mask.shape or values.ndim < 2:
        raise ConvLSTMError("normalization VTEC and mask shapes must match")
    selected = values[mask]
    if selected.numel() < 2:
        raise ConvLSTMError("normalization requires at least two valid cells")
    if not _torch.isfinite(selected).all().item():
        raise ConvLSTMError("valid normalization cells must be finite")
    center = selected.median()
    mad = (selected - center).abs().median()
    scale = max(float(mad.item()) * 1.4826, float(scale_floor_tecu))
    return GridNormalization(center_tecu=float(center.item()), scale_tecu=scale)


def build_causal_batch(
    vtec_tecu: Any,
    valid_mask: Any,
    target_indices: Iterable[int],
    *,
    config: ConvLSTMConfig,
    normalization: GridNormalization,
    observed_at: Sequence[datetime] | None = None,
) -> CausalGridBatch:
    """Construct supervised windows without placing targets in model inputs.

    ``vtec_tecu`` and ``valid_mask`` must have shape ``[time, latitude,
    longitude]``.  Invalid VTEC cells are replaced with normalized zero and
    accompanied by a zero mask channel.  Loss is permitted only where both
    the next-frame target and the prior-day reference are valid.
    """

    _require_torch()
    values = _torch.as_tensor(vtec_tecu, dtype=_torch.float32, device="cpu")
    mask = _torch.as_tensor(valid_mask, dtype=_torch.bool, device="cpu")
    _validate_grid_series(values, mask, config)
    if observed_at is not None:
        _validate_observed_times(observed_at, len(values), config.cadence_seconds)

    indices = tuple(target_indices)
    if not indices:
        raise ConvLSTMError("at least one target index is required")
    if any(
        not isinstance(index, int) or isinstance(index, bool) for index in indices
    ):
        raise ConvLSTMError("target indices must be integers")
    if len(set(indices)) != len(indices):
        raise ConvLSTMError("target indices must be unique")

    inputs: list[Any] = []
    references: list[Any] = []
    target_residuals: list[Any] = []
    targets: list[Any] = []
    loss_masks: list[Any] = []
    center = float(normalization.center_tecu)
    scale = float(normalization.scale_tecu)
    for target_index in indices:
        history_start = target_index - config.history_frames
        reference_index = target_index - config.reference_lag_frames
        if history_start < 0 or target_index >= len(values):
            raise ConvLSTMError(
                "target index does not have a complete causal history window"
            )
        history_values = values[history_start:target_index]
        history_mask = mask[history_start:target_index]
        selected_values = _torch.where(
            history_mask,
            (history_values - center) / scale,
            _torch.zeros_like(history_values),
        )
        inputs.append(
            _torch.stack((selected_values, history_mask.to(_torch.float32)), dim=1)
        )
        reference = values[reference_index]
        target = values[target_index]
        target_mask = mask[target_index] & mask[reference_index]
        references.append(reference.unsqueeze(0))
        targets.append(target.unsqueeze(0))
        target_residuals.append(((target - reference) / scale).unsqueeze(0))
        loss_masks.append(target_mask.unsqueeze(0))

    return CausalGridBatch(
        inputs=_torch.stack(inputs),
        reference_vtec=_torch.stack(references),
        target_residual=_torch.stack(target_residuals),
        target_vtec=_torch.stack(targets),
        loss_mask=_torch.stack(loss_masks),
        target_indices=indices,
    )


def create_model(
    config: ConvLSTMConfig | None = None,
    *,
    device: str = "cpu",
) -> Any:
    """Create deterministically initialized model parameters on the CPU."""

    _require_torch()
    _require_cpu_device(device)
    selected = config or ConvLSTMConfig()
    # Avoid mutating the application's process-wide RNG state merely by
    # constructing a baseline model.
    with _torch.random.fork_rng(devices=[]):
        _torch.manual_seed(selected.initialization_seed)
        model = PredictiveConvLSTM(selected)
    return model.to(device="cpu", dtype=_torch.float32)


def create_optimizer(model: Any, config: ConvLSTMConfig | None = None) -> Any:
    """Create the fixed AdamW optimizer used by resumable training."""

    _require_torch()
    selected = config or getattr(model, "config", None)
    if not isinstance(selected, ConvLSTMConfig):
        raise ConvLSTMError("optimizer requires a ConvLSTMConfig")
    _require_matching_model(model, selected)
    # Explicit for-loop AdamW avoids backend-dependent foreach selection.
    return _torch.optim.AdamW(
        model.parameters(),
        lr=selected.learning_rate,
        weight_decay=selected.weight_decay,
        foreach=False,
    )


def predict_next(
    model: Any,
    history_vtec_tecu: Any,
    history_valid_mask: Any,
    *,
    config: ConvLSTMConfig,
    normalization: GridNormalization,
) -> GridPrediction:
    """Predict one next map using history only; no target argument exists."""

    _require_torch()
    _require_matching_model(model, config)
    values = _torch.as_tensor(
        history_vtec_tecu,
        dtype=_torch.float32,
        device="cpu",
    )
    mask = _torch.as_tensor(history_valid_mask, dtype=_torch.bool, device="cpu")
    if values.ndim == 3:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    expected = (
        config.history_frames,
        config.grid_height,
        config.grid_width,
    )
    if values.ndim != 4 or tuple(values.shape[1:]) != expected:
        raise ConvLSTMError(
            "prediction history must have shape [B,T,H,W] for the configuration"
        )
    if values.shape != mask.shape:
        raise ConvLSTMError("prediction history VTEC and mask shapes must match")
    if not _torch.isfinite(values[mask]).all().item():
        raise ConvLSTMError("valid prediction history cells must be finite")
    normalized = _torch.where(
        mask,
        (values - normalization.center_tecu) / normalization.scale_tecu,
        _torch.zeros_like(values),
    )
    inputs = _torch.stack((normalized, mask.to(_torch.float32)), dim=2)
    reference_index = config.reference_history_index
    reference = values[:, reference_index].unsqueeze(1)
    reference_valid = mask[:, reference_index].unsqueeze(1)
    model.eval()
    with _torch.no_grad():
        normalized_residual = model(inputs)
    residual = normalized_residual * normalization.scale_tecu
    return GridPrediction(
        predicted_vtec_tecu=(reference + residual).detach().cpu(),
        predicted_residual_tecu=residual.detach().cpu(),
        valid_mask=reference_valid.detach().cpu(),
    )


def masked_l1_loss(prediction: Any, target: Any, valid_mask: Any) -> Any:
    """Mean absolute error over valid target/reference cells only."""

    _require_torch()
    if not _torch.is_tensor(prediction) or not _torch.is_tensor(target):
        raise ConvLSTMError("loss prediction and target must be tensors")
    if prediction.shape != target.shape:
        raise ConvLSTMError("loss prediction and target shapes must match")
    mask = _torch.as_tensor(valid_mask, dtype=_torch.bool, device=prediction.device)
    if mask.shape != prediction.shape:
        raise ConvLSTMError("loss mask shape must match prediction")
    if not _torch.isfinite(prediction).all().item():
        raise ConvLSTMError("loss prediction contains non-finite values")
    if not _torch.isfinite(target[mask]).all().item():
        raise ConvLSTMError("valid loss targets must be finite")
    count = mask.sum()
    if int(count.item()) == 0:
        raise ConvLSTMError("masked loss requires at least one valid cell")
    absolute_error = (prediction - target).abs()
    # IEEE NaN multiplied by zero is still NaN, so select rather than multiply
    # when missing source cells contain NaN sentinels.
    selected_error = _torch.where(
        mask,
        absolute_error,
        _torch.zeros_like(absolute_error),
    )
    return selected_error.sum() / count


def train_batch(model: Any, optimizer: Any, batch: CausalGridBatch) -> float:
    """Run one deterministic CPU optimization step and return masked L1."""

    _require_torch()
    config = getattr(model, "config", None)
    if not isinstance(config, ConvLSTMConfig):
        raise ConvLSTMError("training model does not expose its configuration")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.inputs.to(device="cpu", dtype=_torch.float32))
    loss = masked_l1_loss(
        prediction,
        batch.target_residual.to(device="cpu", dtype=_torch.float32),
        batch.loss_mask.to(device="cpu"),
    )
    loss.backward()
    _torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu().item())


def train_epoch(
    model: Any,
    optimizer: Any,
    batches: Iterable[CausalGridBatch],
) -> float:
    """Train ordered batches once, returning their unweighted mean loss."""

    losses = [train_batch(model, optimizer, batch) for batch in batches]
    if not losses:
        raise ConvLSTMError("training epoch requires at least one batch")
    return sum(losses) / len(losses)


def serialize_model(
    model: Any,
    *,
    config: ConvLSTMConfig,
    normalization: GridNormalization,
    training_data_sha256: str,
    completed_epochs: int,
) -> bytes:
    """Serialize a deterministic, pickle-free inference artifact."""

    _require_torch()
    _require_matching_model(model, config)
    _validate_digest(training_data_sha256, "training_data_sha256")
    _nonnegative_integer(completed_epochs, "completed_epochs")
    payload = _model_payload(
        model,
        config=config,
        normalization=normalization,
        training_data_sha256=training_data_sha256,
        completed_epochs=completed_epochs,
    )
    payload["artifact_format"] = ARTIFACT_FORMAT
    return _canonical_json(payload)


def load_model(content: bytes, *, device: str = "cpu") -> LoadedConvLSTMModel:
    """Restore a canonical model artifact strictly onto CPU."""

    _require_torch()
    _require_cpu_device(device)
    payload = _parse_canonical_json(content, maximum_bytes=_MAX_SERIALIZED_BYTES)
    _validate_root(payload, ARTIFACT_FORMAT, include_optimizer=False)
    config, normalization, digest, epochs = _metadata_from_payload(payload)
    model = create_model(config)
    _load_module_state(model, payload["model_state"])
    if (
        serialize_model(
            model,
            config=config,
            normalization=normalization,
            training_data_sha256=digest,
            completed_epochs=epochs,
        )
        != content
    ):
        raise ConvLSTMArtifactError("model artifact is not canonically encoded")
    model.eval()
    return LoadedConvLSTMModel(
        model=model,
        config=config,
        normalization=normalization,
        training_data_sha256=digest,
        completed_epochs=epochs,
        artifact_sha256=hashlib.sha256(content).hexdigest(),
    )


def serialize_checkpoint(
    model: Any,
    optimizer: Any,
    *,
    config: ConvLSTMConfig,
    normalization: GridNormalization,
    training_data_sha256: str,
    completed_epochs: int,
    next_window_index: int,
) -> bytes:
    """Save exact model/AdamW state and the durable training cursor."""

    _require_torch()
    _require_matching_model(model, config)
    if not isinstance(optimizer, _torch.optim.AdamW):
        raise ConvLSTMError("checkpoint optimizer must be torch.optim.AdamW")
    _validate_digest(training_data_sha256, "training_data_sha256")
    _nonnegative_integer(completed_epochs, "completed_epochs")
    _nonnegative_integer(next_window_index, "next_window_index")
    payload = _model_payload(
        model,
        config=config,
        normalization=normalization,
        training_data_sha256=training_data_sha256,
        completed_epochs=completed_epochs,
    )
    payload["artifact_format"] = CHECKPOINT_FORMAT
    payload["training"]["next_window_index"] = next_window_index
    payload["optimizer"] = {
        "algorithm": "torch.optim.AdamW",
        "state": _encode_tree(optimizer.state_dict()),
    }
    return _canonical_json(payload)


def load_checkpoint(
    content: bytes,
    *,
    device: str = "cpu",
) -> RestoredTrainingState:
    """Restore model, AdamW moments, and the next-window cursor onto CPU."""

    _require_torch()
    _require_cpu_device(device)
    payload = _parse_canonical_json(content, maximum_bytes=_MAX_SERIALIZED_BYTES)
    _validate_root(payload, CHECKPOINT_FORMAT, include_optimizer=True)
    config, normalization, digest, epochs = _metadata_from_payload(payload)
    training = payload["training"]
    if set(training) != {
        "completed_epochs",
        "data_sha256",
        "next_window_index",
    }:
        raise ConvLSTMArtifactError("checkpoint training metadata is invalid")
    next_window = training["next_window_index"]
    _nonnegative_integer(next_window, "next_window_index", artifact=True)
    optimizer_payload = payload["optimizer"]
    if (
        not isinstance(optimizer_payload, dict)
        or set(optimizer_payload) != {"algorithm", "state"}
        or optimizer_payload.get("algorithm") != "torch.optim.AdamW"
    ):
        raise ConvLSTMArtifactError("checkpoint optimizer metadata is invalid")

    model = create_model(config)
    _load_module_state(model, payload["model_state"])
    optimizer = create_optimizer(model, config)
    try:
        optimizer.load_state_dict(_decode_tree(optimizer_payload["state"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ConvLSTMArtifactError("checkpoint optimizer state is invalid") from error
    if (
        serialize_checkpoint(
            model,
            optimizer,
            config=config,
            normalization=normalization,
            training_data_sha256=digest,
            completed_epochs=epochs,
            next_window_index=next_window,
        )
        != content
    ):
        raise ConvLSTMArtifactError("checkpoint is not canonically encoded")
    return RestoredTrainingState(
        model=model,
        optimizer=optimizer,
        config=config,
        normalization=normalization,
        training_data_sha256=digest,
        completed_epochs=epochs,
        next_window_index=next_window,
        checkpoint_sha256=hashlib.sha256(content).hexdigest(),
    )


def _model_payload(
    model: Any,
    *,
    config: ConvLSTMConfig,
    normalization: GridNormalization,
    training_data_sha256: str,
    completed_epochs: int,
) -> dict[str, Any]:
    return {
        "algorithm": ALGORITHM,
        "configuration": config.to_dict(),
        "model_state": {
            name: _encode_tensor(tensor)
            for name, tensor in sorted(model.state_dict().items())
        },
        "normalization": normalization.to_dict(),
        "training": {
            "completed_epochs": completed_epochs,
            "data_sha256": training_data_sha256,
        },
    }


def _metadata_from_payload(
    payload: Mapping[str, Any],
) -> tuple[ConvLSTMConfig, GridNormalization, str, int]:
    configuration = payload["configuration"]
    if not isinstance(configuration, dict):
        raise ConvLSTMArtifactError("artifact configuration must be an object")
    config = ConvLSTMConfig.from_bytes(_canonical_json(configuration))
    normalization_payload = payload["normalization"]
    if (
        not isinstance(normalization_payload, dict)
        or set(normalization_payload) != {"center_tecu", "scale_tecu"}
    ):
        raise ConvLSTMArtifactError("artifact normalization is invalid")
    try:
        normalization = GridNormalization(**normalization_payload)
    except (TypeError, ValueError) as error:
        raise ConvLSTMArtifactError("artifact normalization is invalid") from error
    training = payload["training"]
    if not isinstance(training, dict):
        raise ConvLSTMArtifactError("artifact training metadata is invalid")
    digest = training.get("data_sha256")
    epochs = training.get("completed_epochs")
    _validate_digest(digest, "training data hash", artifact=True)
    _nonnegative_integer(epochs, "completed_epochs", artifact=True)
    return config, normalization, digest, epochs


def _validate_root(
    payload: Any,
    artifact_format: str,
    *,
    include_optimizer: bool,
) -> None:
    if not isinstance(payload, dict):
        raise ConvLSTMArtifactError("artifact root must be an object")
    expected = {
        "algorithm",
        "artifact_format",
        "configuration",
        "model_state",
        "normalization",
        "training",
    }
    if include_optimizer:
        expected.add("optimizer")
    if set(payload) != expected:
        raise ConvLSTMArtifactError("artifact fields do not match the contract")
    if payload.get("algorithm") != ALGORITHM:
        raise ConvLSTMArtifactError("unsupported ConvLSTM algorithm")
    if payload.get("artifact_format") != artifact_format:
        raise ConvLSTMArtifactError("unsupported ConvLSTM artifact format")
    if not isinstance(payload.get("model_state"), dict):
        raise ConvLSTMArtifactError("artifact model state must be an object")
    if not include_optimizer and set(payload["training"]) != {
        "completed_epochs",
        "data_sha256",
    }:
        raise ConvLSTMArtifactError("model training metadata is invalid")


def _load_module_state(model: Any, encoded_state: Mapping[str, Any]) -> None:
    expected = model.state_dict()
    if set(encoded_state) != set(expected):
        raise ConvLSTMArtifactError("artifact model parameters do not match architecture")
    restored: dict[str, Any] = {}
    for name, expected_tensor in expected.items():
        tensor = _decode_tensor(encoded_state[name])
        if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
            raise ConvLSTMArtifactError(
                f"artifact tensor {name!r} does not match architecture"
            )
        restored[name] = tensor
    try:
        model.load_state_dict(restored, strict=True)
    except RuntimeError as error:
        raise ConvLSTMArtifactError("artifact model state cannot be loaded") from error


def _encode_tensor(tensor: Any) -> dict[str, Any]:
    _require_torch()
    if not _torch.is_tensor(tensor):
        raise ConvLSTMArtifactError("only tensors may appear in tensor state")
    value = tensor.detach().to(device="cpu").contiguous()
    dtype_name = str(value.dtype).removeprefix("torch.")
    if dtype_name not in _SUPPORTED_TENSOR_DTYPES:
        raise ConvLSTMArtifactError(f"unsupported tensor dtype {dtype_name!r}")
    if value.is_floating_point() and not _torch.isfinite(value).all().item():
        raise ConvLSTMArtifactError("tensor state contains non-finite values")
    # Do not make the artifact contract depend on NumPy: PyTorch is the sole
    # optional runtime required by this module.  Models are deliberately
    # compact, so materializing their byte view as a Python byte sequence is a
    # reasonable serialization trade-off.
    raw = bytes(value.reshape(-1).view(_torch.uint8).tolist())
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "dtype": dtype_name,
        "shape": list(value.shape),
    }


def _decode_tensor(payload: Any) -> Any:
    _require_torch()
    if (
        not isinstance(payload, dict)
        or set(payload) != {"data_base64", "dtype", "shape"}
    ):
        raise ConvLSTMArtifactError("encoded tensor fields are invalid")
    dtype_name = payload["dtype"]
    if dtype_name not in _SUPPORTED_TENSOR_DTYPES:
        raise ConvLSTMArtifactError("encoded tensor dtype is unsupported")
    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > 8
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in shape
        )
    ):
        raise ConvLSTMArtifactError("encoded tensor shape is invalid")
    element_count = math.prod(shape)
    if element_count > 32_000_000:
        raise ConvLSTMArtifactError("encoded tensor is too large")
    data = payload["data_base64"]
    if not isinstance(data, str):
        raise ConvLSTMArtifactError("encoded tensor data must be text")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ConvLSTMArtifactError("encoded tensor base64 is invalid") from error
    dtype = getattr(_torch, _SUPPORTED_TENSOR_DTYPES[dtype_name])
    element_size = _torch.empty((), dtype=dtype).element_size()
    if len(raw) != element_count * element_size:
        raise ConvLSTMArtifactError("encoded tensor byte length is invalid")
    # bytearray gives frombuffer owned writable storage; clone disconnects the
    # returned tensor from this short-lived serialization buffer.
    tensor = _torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
    if tensor.is_floating_point() and not _torch.isfinite(tensor).all().item():
        raise ConvLSTMArtifactError("encoded tensor contains non-finite values")
    return tensor


def _encode_tree(value: Any) -> Any:
    if TORCH_AVAILABLE and _torch.is_tensor(value):
        return {"kind": "tensor", "value": _encode_tensor(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConvLSTMArtifactError("optimizer state contains non-finite values")
        return value
    if isinstance(value, Mapping):
        items = [(_encode_tree(key), _encode_tree(item)) for key, item in value.items()]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"items": [[key, item] for key, item in items], "kind": "mapping"}
    if isinstance(value, list):
        return {"items": [_encode_tree(item) for item in value], "kind": "list"}
    if isinstance(value, tuple):
        return {"items": [_encode_tree(item) for item in value], "kind": "tuple"}
    raise ConvLSTMArtifactError(
        f"optimizer state contains unsupported {type(value).__name__}"
    )


def _decode_tree(value: Any) -> Any:
    if not isinstance(value, dict) or "kind" not in value:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ConvLSTMArtifactError("encoded optimizer scalar is invalid")
    kind = value.get("kind")
    if kind == "tensor" and set(value) == {"kind", "value"}:
        return _decode_tensor(value["value"])
    if kind in {"list", "tuple"} and set(value) == {"items", "kind"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ConvLSTMArtifactError("encoded optimizer sequence is invalid")
        decoded = [_decode_tree(item) for item in items]
        return decoded if kind == "list" else tuple(decoded)
    if kind == "mapping" and set(value) == {"items", "kind"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ConvLSTMArtifactError("encoded optimizer mapping is invalid")
        result: dict[Any, Any] = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ConvLSTMArtifactError("encoded optimizer mapping pair is invalid")
            key = _decode_tree(pair[0])
            try:
                if key in result:
                    raise ConvLSTMArtifactError(
                        "encoded optimizer mapping has duplicate keys"
                    )
                result[key] = _decode_tree(pair[1])
            except TypeError as error:
                raise ConvLSTMArtifactError(
                    "encoded optimizer mapping key is invalid"
                ) from error
        return result
    raise ConvLSTMArtifactError("encoded optimizer value is invalid")


def _validate_grid_series(values: Any, mask: Any, config: ConvLSTMConfig) -> None:
    expected_tail = (config.grid_height, config.grid_width)
    if values.ndim != 3 or tuple(values.shape[1:]) != expected_tail:
        raise ConvLSTMError(
            "VTEC series must have shape [time,grid_height,grid_width]"
        )
    if values.shape != mask.shape:
        raise ConvLSTMError("VTEC and valid-mask shapes must match")
    if not _torch.isfinite(values[mask]).all().item():
        raise ConvLSTMError("valid VTEC cells must be finite")


def _validate_observed_times(
    observed_at: Sequence[datetime],
    count: int,
    cadence_seconds: int,
) -> None:
    if len(observed_at) != count:
        raise ConvLSTMError("observed_at length must match the grid time axis")
    previous: datetime | None = None
    for instant in observed_at:
        if not isinstance(instant, datetime) or instant.tzinfo is None:
            raise ConvLSTMError("observed_at values must be timezone-aware datetimes")
        current = instant.astimezone(UTC)
        if previous is not None:
            seconds = (current - previous).total_seconds()
            if abs(seconds - cadence_seconds) > 1e-6:
                raise ConvLSTMError(
                    "observed_at values must follow the configured cadence"
                )
        previous = current


def _require_matching_model(model: Any, config: ConvLSTMConfig) -> None:
    if not isinstance(model, PredictiveConvLSTM) or model.config != config:
        raise ConvLSTMError("model architecture does not match the configuration")
    for parameter in model.parameters():
        if parameter.device.type != "cpu":
            raise ConvLSTMError("this reference pipeline supports CPU models only")


def _require_cpu_device(device: str) -> None:
    if str(device) not in {"cpu", "cpu:0"}:
        raise ConvLSTMError("this reference pipeline supports device='cpu' only")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ConvLSTMUnavailableError(
            "ConvLSTM operations require the optional 'spatial' dependency "
            "(install ophanim[spatial])"
        )


def _canonical_json(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConvLSTMArtifactError("value cannot be encoded as canonical JSON") from error


def _parse_canonical_json(content: bytes, *, maximum_bytes: int) -> Any:
    if not isinstance(content, bytes):
        raise TypeError("serialized ConvLSTM content must be bytes")
    if not content or len(content) > maximum_bytes:
        raise ConvLSTMArtifactError("serialized ConvLSTM content size is invalid")
    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda value: (_raise_nonfinite_json(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConvLSTMArtifactError("serialized content is not valid UTF-8 JSON") from error
    if _canonical_json(payload) != content:
        raise ConvLSTMArtifactError("serialized content is not canonical JSON")
    return payload


def _raise_nonfinite_json(value: str) -> Any:
    raise ConvLSTMArtifactError(f"non-finite JSON number {value!r} is forbidden")


def _validate_digest(value: Any, name: str, *, artifact: bool = False) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        error = ConvLSTMArtifactError if artifact else ConvLSTMError
        raise error(f"{name} must be a lowercase SHA-256 digest")


def _nonnegative_integer(value: Any, name: str, *, artifact: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        error = ConvLSTMArtifactError if artifact else ConvLSTMError
        raise error(f"{name} must be a non-negative integer")
