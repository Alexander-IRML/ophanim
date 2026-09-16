"""Small, dependency-free selective state-space forecasting reference.

This module is deliberately not a wrapper around the CUDA-oriented
``mamba-ssm`` package.  It implements the defining selective state-space
recurrence directly in ordinary Python so OPHANIM has a deterministic CPU
baseline that can be exercised on any supported laptop.

The recurrent feature extractor is fixed and versioned.  Its transition step
(``delta``), input projection (``B``), and output gate (``C``) are functions of
the current input.  A two-output ridge readout is then fitted to predict the
next regional mean and median VTEC.  Held-out, chronologically later residuals
provide robust median/MAD anomaly calibration.
"""

from __future__ import annotations

import calendar
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from statistics import median
from typing import Any, ClassVar, Self


ALGORITHM = "ophanim-mamba-selective-ssm/1"
ARTIFACT_FORMAT = "ophanim-mamba-model/1"
FEATURE_NAMES = (
    "log1p_mean_vtec",
    "log1p_median_vtec",
    "coverage",
    "utc_hour_sin",
    "utc_hour_cos",
    "year_phase_sin",
    "year_phase_cos",
)
_FEATURE_COUNT = len(FEATURE_NAMES)
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
# Synthetic gap-fill samples use tiny, nonzero coverage so they remain valid
# recurrent inputs.  They preserve elapsed cadence/state, but are not observed
# targets and therefore must not influence the fitted readout or calibration.
_SUPERVISED_TARGET_MIN_COVERAGE = 1e-5


class MambaModelError(ValueError):
    """The requested training or scoring operation is not well-defined."""


class MambaArtifactError(MambaModelError):
    """Serialized model bytes do not satisfy the versioned artifact contract."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One timestamped regional readout used for fitting or scoring."""

    observed_at: datetime
    mean_vtec_tecu: float
    median_vtec_tecu: float
    coverage_fraction: float = 1.0

    def __post_init__(self) -> None:
        _aware(self.observed_at, "sample observed_at")
        for name, value in (
            ("mean_vtec_tecu", self.mean_vtec_tecu),
            ("median_vtec_tecu", self.median_vtec_tecu),
            ("coverage_fraction", self.coverage_fraction),
        ):
            _finite_number(value, f"sample {name}")
        if self.mean_vtec_tecu < 0.0 or self.median_vtec_tecu < 0.0:
            raise MambaModelError("sample VTEC values must be non-negative")
        if not 0.0 < self.coverage_fraction <= 1.0:
            raise MambaModelError("sample coverage_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe canonical representation of the sample."""

        return {
            "coverage_fraction": float(self.coverage_fraction),
            "mean_vtec_tecu": float(self.mean_vtec_tecu),
            "median_vtec_tecu": float(self.median_vtec_tecu),
            "observed_at": _datetime_text(self.observed_at),
        }


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One next-readout forecast and its calibrated anomaly assessment."""

    observed_at: datetime
    predicted_mean_vtec_tecu: float
    predicted_median_vtec_tecu: float
    observed_mean_vtec_tecu: float
    observed_median_vtec_tecu: float
    mean_residual_tecu: float
    median_residual_tecu: float
    mean_anomaly_score: float
    median_anomaly_score: float
    anomaly_score: float
    threshold: float
    mean_is_candidate: bool
    median_is_candidate: bool
    is_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a response payload containing no enums or datetime objects."""

        return {
            "anomaly_score": self.anomaly_score,
            "is_candidate": self.is_candidate,
            "mean_anomaly_score": self.mean_anomaly_score,
            "mean_is_candidate": self.mean_is_candidate,
            "mean_residual_tecu": self.mean_residual_tecu,
            "median_anomaly_score": self.median_anomaly_score,
            "median_is_candidate": self.median_is_candidate,
            "median_residual_tecu": self.median_residual_tecu,
            "observed_at": _datetime_text(self.observed_at),
            "observed_mean_vtec_tecu": self.observed_mean_vtec_tecu,
            "observed_median_vtec_tecu": self.observed_median_vtec_tecu,
            "predicted_mean_vtec_tecu": self.predicted_mean_vtec_tecu,
            "predicted_median_vtec_tecu": self.predicted_median_vtec_tecu,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class MambaModel:
    """Immutable parameters for OPHANIM's CPU selective-SSM reference.

    ``anchor_state`` is the recurrent state *after* the final initialization
    sample.  Consequently :meth:`score_sequence` can begin with the first new
    readout without replaying twenty years of history.  Optional ``history``
    consists only of already-seen samples after that anchor, such as readouts
    scored during an earlier button press.
    """

    algorithm: ClassVar[str] = ALGORITHM
    artifact_format: ClassVar[str] = ARTIFACT_FORMAT

    threshold: float
    state_size: int
    ridge_penalty: float
    cadence_seconds: float
    residual_scale_floor_tecu: float
    input_centers: tuple[float, float]
    input_scales: tuple[float, float]
    decay_rates: tuple[float, ...]
    delta_biases: tuple[float, ...]
    delta_weights: tuple[tuple[float, ...], ...]
    input_biases: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    output_biases: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    mean_readout: tuple[float, ...]
    median_readout: tuple[float, ...]
    mean_residual_center_tecu: float
    mean_residual_scale_tecu: float
    median_residual_center_tecu: float
    median_residual_scale_tecu: float
    training_start_at: datetime
    readout_fit_end_at: datetime
    training_end_at: datetime
    training_sample_count: int
    calibration_sample_count: int
    training_data_sha256: str
    anchor_sample: TrainingSample
    anchor_state: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("threshold", self.threshold),
            ("ridge_penalty", self.ridge_penalty),
            ("cadence_seconds", self.cadence_seconds),
            ("residual_scale_floor_tecu", self.residual_scale_floor_tecu),
            ("mean_residual_center_tecu", self.mean_residual_center_tecu),
            ("mean_residual_scale_tecu", self.mean_residual_scale_tecu),
            ("median_residual_center_tecu", self.median_residual_center_tecu),
            ("median_residual_scale_tecu", self.median_residual_scale_tecu),
        ):
            _finite_number(value, f"model {name}")
        if self.threshold <= 0.0:
            raise MambaArtifactError("model threshold must be positive")
        if self.ridge_penalty <= 0.0:
            raise MambaArtifactError("model ridge_penalty must be positive")
        if self.cadence_seconds <= 0.0:
            raise MambaArtifactError("model cadence_seconds must be positive")
        if self.residual_scale_floor_tecu <= 0.0:
            raise MambaArtifactError(
                "model residual_scale_floor_tecu must be positive"
            )
        if self.mean_residual_scale_tecu < self.residual_scale_floor_tecu or (
            self.median_residual_scale_tecu < self.residual_scale_floor_tecu
        ):
            raise MambaArtifactError(
                "model residual scales must respect their configured floor"
            )
        if (
            not isinstance(self.state_size, int)
            or isinstance(self.state_size, bool)
            or self.state_size < 1
            or self.state_size > 256
        ):
            raise MambaArtifactError("model state_size must be in [1, 256]")
        for name, value in (
            ("training_sample_count", self.training_sample_count),
            ("calibration_sample_count", self.calibration_sample_count),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise MambaArtifactError(f"model {name} must be a positive integer")
        if self.calibration_sample_count >= self.training_sample_count:
            raise MambaArtifactError(
                "calibration sample count must be smaller than training sample count"
            )
        _validate_digest(self.training_data_sha256, "training_data_sha256")
        for name, instant in (
            ("training_start_at", self.training_start_at),
            ("readout_fit_end_at", self.readout_fit_end_at),
            ("training_end_at", self.training_end_at),
        ):
            _aware(instant, f"model {name}")
        if not (
            self.training_start_at
            <= self.readout_fit_end_at
            < self.training_end_at
        ):
            raise MambaArtifactError(
                "model training timestamps are not chronologically ordered"
            )
        if self.anchor_sample.observed_at != self.training_end_at:
            raise MambaArtifactError(
                "model anchor sample must be the final initialization sample"
            )

        _validate_vector(self.input_centers, 2, "input_centers")
        _validate_vector(self.input_scales, 2, "input_scales", positive=True)
        for name, vector in (
            ("decay_rates", self.decay_rates),
            ("delta_biases", self.delta_biases),
            ("input_biases", self.input_biases),
            ("output_biases", self.output_biases),
            ("anchor_state", self.anchor_state),
        ):
            _validate_vector(
                vector,
                self.state_size,
                name,
                positive=name == "decay_rates",
            )
        for name, matrix in (
            ("delta_weights", self.delta_weights),
            ("input_weights", self.input_weights),
            ("output_weights", self.output_weights),
        ):
            if not isinstance(matrix, tuple) or len(matrix) != self.state_size:
                raise MambaArtifactError(
                    f"model {name} must have state_size rows"
                )
            for row in matrix:
                _validate_vector(row, _FEATURE_COUNT, f"{name} row")
        readout_size = 1 + _FEATURE_COUNT + self.state_size
        _validate_vector(self.mean_readout, readout_size, "mean_readout")
        _validate_vector(self.median_readout, readout_size, "median_readout")

    @property
    def artifact_sha256(self) -> str:
        """SHA-256 identity of the exact canonical artifact bytes."""

        return sha256(self.to_bytes()).hexdigest()

    @property
    def configuration_sha256(self) -> str:
        """Hash model behavior independently of training data and weights."""

        return sha256(_canonical_json(self._configuration_payload())).hexdigest()

    def score_sequence(
        self,
        history: Iterable[TrainingSample],
        targets: Iterable[TrainingSample],
    ) -> tuple[ScoreResult, ...]:
        """Score new chronological readouts without looking at their values.

        Prediction for each target is calculated from the state ending at the
        preceding sample.  Only after the forecast is captured is the target's
        observed value passed into the recurrence.  ``history`` may replay
        already-processed readouts after ``training_end_at``; it must not
        overlap the initialization data encoded by ``anchor_state``.
        """

        history_values = _ordered_samples(history, name="history", allow_empty=True)
        target_values = _ordered_samples(targets, name="targets", allow_empty=True)
        if not target_values:
            return ()

        state = self.anchor_state
        previous = self.anchor_sample
        if history_values and history_values[0].observed_at <= self.training_end_at:
            raise MambaModelError(
                "scoring history must occur after the model initialization data"
            )
        for sample in history_values:
            self._require_next_readout(previous.observed_at, sample.observed_at)
            state = self._advance(state, sample, previous.observed_at)
            previous = sample
        if target_values[0].observed_at <= previous.observed_at:
            raise MambaModelError(
                "scoring targets must occur after initialization and history"
            )

        results: list[ScoreResult] = []
        for target in target_values:
            self._require_next_readout(previous.observed_at, target.observed_at)

            # Deliberately forecast before target is supplied to the recurrence.
            row = self._readout_row(state, previous)
            predicted_mean = self._decode_target(
                _dot(self.mean_readout, row),
                metric_index=0,
            )
            predicted_median = self._decode_target(
                _dot(self.median_readout, row),
                metric_index=1,
            )
            mean_residual = target.mean_vtec_tecu - predicted_mean
            median_residual = target.median_vtec_tecu - predicted_median
            mean_score = abs(
                (mean_residual - self.mean_residual_center_tecu)
                / self.mean_residual_scale_tecu
            )
            median_score = abs(
                (median_residual - self.median_residual_center_tecu)
                / self.median_residual_scale_tecu
            )
            mean_candidate = mean_score >= self.threshold
            median_candidate = median_score >= self.threshold
            results.append(
                ScoreResult(
                    observed_at=target.observed_at.astimezone(UTC),
                    predicted_mean_vtec_tecu=predicted_mean,
                    predicted_median_vtec_tecu=predicted_median,
                    observed_mean_vtec_tecu=target.mean_vtec_tecu,
                    observed_median_vtec_tecu=target.median_vtec_tecu,
                    mean_residual_tecu=mean_residual,
                    median_residual_tecu=median_residual,
                    mean_anomaly_score=mean_score,
                    median_anomaly_score=median_score,
                    anomaly_score=max(mean_score, median_score),
                    threshold=self.threshold,
                    mean_is_candidate=mean_candidate,
                    median_is_candidate=median_candidate,
                    is_candidate=mean_candidate or median_candidate,
                )
            )

            state = self._advance(state, target, previous.observed_at)
            previous = target
        return tuple(results)

    def to_bytes(self) -> bytes:
        """Serialize as deterministic, finite-number-only canonical JSON."""

        payload = {
            "algorithm": ALGORITHM,
            "anchor": {
                "sample": self.anchor_sample.to_dict(),
                "state": list(self.anchor_state),
            },
            "artifact_format": ARTIFACT_FORMAT,
            "calibration": {
                "mean_residual_center_tecu": self.mean_residual_center_tecu,
                "mean_residual_scale_tecu": self.mean_residual_scale_tecu,
                "median_residual_center_tecu": self.median_residual_center_tecu,
                "median_residual_scale_tecu": self.median_residual_scale_tecu,
                "sample_count": self.calibration_sample_count,
            },
            "configuration": self._configuration_payload(),
            "input_scaling": {
                "centers": list(self.input_centers),
                "scales": list(self.input_scales),
            },
            "readout": {
                "mean": list(self.mean_readout),
                "median": list(self.median_readout),
            },
            "recurrence": {
                "decay_rates": list(self.decay_rates),
                "delta_biases": list(self.delta_biases),
                "delta_weights": [list(row) for row in self.delta_weights],
                "input_biases": list(self.input_biases),
                "input_weights": [list(row) for row in self.input_weights],
                "output_biases": list(self.output_biases),
                "output_weights": [list(row) for row in self.output_weights],
            },
            "training": {
                "data_sha256": self.training_data_sha256,
                "readout_fit_end_at": _datetime_text(self.readout_fit_end_at),
                "sample_count": self.training_sample_count,
                "start_at": _datetime_text(self.training_start_at),
                "end_at": _datetime_text(self.training_end_at),
            },
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        """Load canonical bytes while rejecting malformed or unknown layouts."""

        if not isinstance(content, bytes):
            raise TypeError("model artifact content must be bytes")
        if not content or len(content) > _MAX_ARTIFACT_BYTES:
            raise MambaArtifactError("model artifact size is invalid")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MambaArtifactError("model artifact is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise MambaArtifactError("model artifact root must be an object")
        if payload.get("artifact_format") != ARTIFACT_FORMAT:
            raise MambaArtifactError("unsupported model artifact format")
        if payload.get("algorithm") != ALGORITHM:
            raise MambaArtifactError("unsupported model algorithm")
        try:
            configuration = _object(payload, "configuration")
            if tuple(configuration["feature_names"]) != FEATURE_NAMES:
                raise MambaArtifactError("model feature contract is unsupported")
            scaling = _object(payload, "input_scaling")
            recurrence = _object(payload, "recurrence")
            readout = _object(payload, "readout")
            calibration_payload = _object(payload, "calibration")
            training = _object(payload, "training")
            anchor = _object(payload, "anchor")
            anchor_sample = _sample_from_payload(_object(anchor, "sample"))
            model = cls(
                threshold=_number(configuration["threshold"], "threshold"),
                state_size=_integer(configuration["state_size"], "state_size"),
                ridge_penalty=_number(
                    configuration["ridge_penalty"], "ridge_penalty"
                ),
                cadence_seconds=_number(
                    configuration["cadence_seconds"], "cadence_seconds"
                ),
                residual_scale_floor_tecu=_number(
                    configuration["residual_scale_floor_tecu"],
                    "residual_scale_floor_tecu",
                ),
                input_centers=_float_tuple(scaling["centers"], "centers"),
                input_scales=_float_tuple(scaling["scales"], "scales"),
                decay_rates=_float_tuple(
                    recurrence["decay_rates"], "decay_rates"
                ),
                delta_biases=_float_tuple(
                    recurrence["delta_biases"], "delta_biases"
                ),
                delta_weights=_float_matrix(
                    recurrence["delta_weights"], "delta_weights"
                ),
                input_biases=_float_tuple(
                    recurrence["input_biases"], "input_biases"
                ),
                input_weights=_float_matrix(
                    recurrence["input_weights"], "input_weights"
                ),
                output_biases=_float_tuple(
                    recurrence["output_biases"], "output_biases"
                ),
                output_weights=_float_matrix(
                    recurrence["output_weights"], "output_weights"
                ),
                mean_readout=_float_tuple(readout["mean"], "mean readout"),
                median_readout=_float_tuple(
                    readout["median"], "median readout"
                ),
                mean_residual_center_tecu=_number(
                    calibration_payload["mean_residual_center_tecu"],
                    "mean residual center",
                ),
                mean_residual_scale_tecu=_number(
                    calibration_payload["mean_residual_scale_tecu"],
                    "mean residual scale",
                ),
                median_residual_center_tecu=_number(
                    calibration_payload["median_residual_center_tecu"],
                    "median residual center",
                ),
                median_residual_scale_tecu=_number(
                    calibration_payload["median_residual_scale_tecu"],
                    "median residual scale",
                ),
                training_start_at=_parse_datetime(training["start_at"]),
                readout_fit_end_at=_parse_datetime(
                    training["readout_fit_end_at"]
                ),
                training_end_at=_parse_datetime(training["end_at"]),
                training_sample_count=_integer(
                    training["sample_count"], "training sample_count"
                ),
                calibration_sample_count=_integer(
                    calibration_payload["sample_count"],
                    "calibration sample_count",
                ),
                training_data_sha256=_text(
                    training["data_sha256"], "training data_sha256"
                ),
                anchor_sample=anchor_sample,
                anchor_state=_float_tuple(anchor["state"], "anchor state"),
            )
        except KeyError as error:
            raise MambaArtifactError(
                f"model artifact is missing field {error.args[0]!r}"
            ) from error
        except (TypeError, ValueError) as error:
            if isinstance(error, MambaArtifactError):
                raise
            raise MambaArtifactError("model artifact contains invalid fields") from error
        if model.to_bytes() != content:
            raise MambaArtifactError("model artifact JSON is not canonical")
        return model

    def _configuration_payload(self) -> dict[str, Any]:
        return {
            "cadence_seconds": self.cadence_seconds,
            "feature_names": list(FEATURE_NAMES),
            "residual_scale_floor_tecu": self.residual_scale_floor_tecu,
            "ridge_penalty": self.ridge_penalty,
            "state_size": self.state_size,
            "threshold": self.threshold,
        }

    def _encode_sample(self, sample: TrainingSample) -> tuple[float, ...]:
        instant = sample.observed_at.astimezone(UTC)
        seconds_of_day = (
            instant.hour * 3600
            + instant.minute * 60
            + instant.second
            + instant.microsecond / 1_000_000
        )
        hour_angle = math.tau * seconds_of_day / 86_400.0
        days_in_year = 366 if calendar.isleap(instant.year) else 365
        year_fraction = (
            instant.timetuple().tm_yday - 1 + seconds_of_day / 86_400.0
        ) / days_in_year
        year_angle = math.tau * year_fraction
        return (
            (math.log1p(sample.mean_vtec_tecu) - self.input_centers[0])
            / self.input_scales[0],
            (math.log1p(sample.median_vtec_tecu) - self.input_centers[1])
            / self.input_scales[1],
            sample.coverage_fraction * 2.0 - 1.0,
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(year_angle),
            math.cos(year_angle),
        )

    def _advance(
        self,
        state: tuple[float, ...],
        sample: TrainingSample,
        previous_at: datetime | None,
    ) -> tuple[float, ...]:
        encoded = self._encode_sample(sample)
        step_ratio = 1.0
        if previous_at is not None:
            step_ratio = (
                sample.observed_at - previous_at
            ).total_seconds() / self.cadence_seconds
        next_state: list[float] = []
        for index in range(self.state_size):
            # Selective SSM: delta, B (candidate), and C (used below) all
            # depend on the current input rather than remaining time invariant.
            selective_delta = _softplus(
                self.delta_biases[index]
                + _dot(self.delta_weights[index], encoded)
            )
            effective_delta = min(32.0, max(1e-8, selective_delta * step_ratio))
            decay = math.exp(-self.decay_rates[index] * effective_delta)
            candidate = math.tanh(
                self.input_biases[index]
                + _dot(self.input_weights[index], encoded)
            )
            next_state.append(decay * state[index] + (1.0 - decay) * candidate)
        return tuple(next_state)

    def _readout_row(
        self,
        state: tuple[float, ...],
        previous: TrainingSample,
    ) -> tuple[float, ...]:
        encoded = self._encode_sample(previous)
        selected = tuple(
            state[index]
            * _sigmoid(
                self.output_biases[index]
                + _dot(self.output_weights[index], encoded)
            )
            for index in range(self.state_size)
        )
        return (1.0, *encoded, *selected)

    def _decode_target(self, normalized_log_value: float, *, metric_index: int) -> float:
        bounded = min(20.0, max(-20.0, normalized_log_value))
        log_value = (
            bounded * self.input_scales[metric_index]
            + self.input_centers[metric_index]
        )
        return max(0.0, math.expm1(log_value))

    def _require_next_readout(self, previous_at: datetime, next_at: datetime) -> None:
        difference = (next_at - previous_at).total_seconds()
        tolerance = max(1.0, self.cadence_seconds * 0.05)
        if abs(difference - self.cadence_seconds) > tolerance:
            raise MambaModelError(
                "scoring samples must be consecutive model-cadence readouts"
            )


def fit_model(
    samples: Iterable[TrainingSample],
    *,
    threshold: float = 4.5,
    state_size: int = 8,
    ridge_penalty: float = 0.01,
    calibration_fraction: float = 0.2,
    minimum_calibration_samples: int = 8,
    residual_scale_floor_tecu: float = 0.05,
) -> MambaModel:
    """Fit a deterministic next-step mean/median selective-SSM model.

    The split is chronological.  The early prefix fits input scaling and the
    ridge readout; the later suffix calibrates residual scores.  No random
    initialization, shuffling, or future-target feature is used.  Very-low-
    coverage gap-fill samples advance the recurrence but are excluded as
    supervised and residual-calibration targets and from robust input scaling.
    """

    values = _ordered_samples(samples, name="training samples", allow_empty=False)
    _finite_number(threshold, "threshold")
    _finite_number(ridge_penalty, "ridge_penalty")
    _finite_number(calibration_fraction, "calibration_fraction")
    _finite_number(residual_scale_floor_tecu, "residual_scale_floor_tecu")
    if threshold <= 0.0:
        raise MambaModelError("threshold must be positive")
    if ridge_penalty <= 0.0:
        raise MambaModelError("ridge_penalty must be positive")
    if not 0.05 <= calibration_fraction <= 0.5:
        raise MambaModelError("calibration_fraction must be in [0.05, 0.5]")
    if residual_scale_floor_tecu <= 0.0:
        raise MambaModelError("residual_scale_floor_tecu must be positive")
    if (
        not isinstance(state_size, int)
        or isinstance(state_size, bool)
        or not 1 <= state_size <= 256
    ):
        raise MambaModelError("state_size must be an integer in [1, 256]")
    if (
        not isinstance(minimum_calibration_samples, int)
        or isinstance(minimum_calibration_samples, bool)
        or minimum_calibration_samples < 3
    ):
        raise MambaModelError("minimum_calibration_samples must be at least 3")

    pair_count = len(values) - 1
    usable_pair_indices = tuple(
        index
        for index in range(pair_count)
        if values[index + 1].coverage_fraction
        > _SUPERVISED_TARGET_MIN_COVERAGE
    )
    usable_target_count = len(usable_pair_indices)
    calibration_count = max(
        minimum_calibration_samples,
        int(round(usable_target_count * calibration_fraction)),
    )
    readout_size = 1 + _FEATURE_COUNT + state_size
    fit_target_count = usable_target_count - calibration_count
    minimum_fit_targets = readout_size + 2
    if fit_target_count < minimum_fit_targets:
        raise MambaModelError(
            f"training provides {usable_target_count} usable supervised targets "
            f"with coverage_fraction > {_SUPERVISED_TARGET_MIN_COVERAGE:g}; "
            f"the chronological calibration policy reserves {calibration_count}, "
            f"leaving {max(0, fit_target_count)} for fitting, but this state size "
            f"requires at least {minimum_fit_targets} fit targets"
        )
    fit_pair_indices = usable_pair_indices[:fit_target_count]
    calibration_pair_indices = usable_pair_indices[fit_target_count:]

    cadence_seconds = _median_positive_deltas(values)
    fit_end_index = fit_pair_indices[-1] + 1
    fit_samples = tuple(
        sample
        for sample in values[: fit_end_index + 1]
        if sample.coverage_fraction > _SUPERVISED_TARGET_MIN_COVERAGE
    )
    mean_center, mean_scale = _robust_log_scaling(
        sample.mean_vtec_tecu for sample in fit_samples
    )
    median_center, median_scale = _robust_log_scaling(
        sample.median_vtec_tecu for sample in fit_samples
    )
    recurrence = _reference_recurrence(state_size)

    provisional = MambaModel(
        threshold=float(threshold),
        state_size=state_size,
        ridge_penalty=float(ridge_penalty),
        cadence_seconds=cadence_seconds,
        residual_scale_floor_tecu=float(residual_scale_floor_tecu),
        input_centers=(mean_center, median_center),
        input_scales=(mean_scale, median_scale),
        decay_rates=recurrence[0],
        delta_biases=recurrence[1],
        delta_weights=recurrence[2],
        input_biases=recurrence[3],
        input_weights=recurrence[4],
        output_biases=recurrence[5],
        output_weights=recurrence[6],
        mean_readout=(0.0,) * readout_size,
        median_readout=(0.0,) * readout_size,
        mean_residual_center_tecu=0.0,
        mean_residual_scale_tecu=float(residual_scale_floor_tecu),
        median_residual_center_tecu=0.0,
        median_residual_scale_tecu=float(residual_scale_floor_tecu),
        training_start_at=values[0].observed_at,
        readout_fit_end_at=values[fit_end_index].observed_at,
        training_end_at=values[-1].observed_at,
        training_sample_count=len(values),
        calibration_sample_count=calibration_count,
        training_data_sha256=_training_data_hash(values),
        anchor_sample=values[-1],
        anchor_state=(0.0,) * state_size,
    )

    rows: list[tuple[float, ...]] = []
    mean_targets: list[float] = []
    median_targets: list[float] = []
    state = (0.0,) * state_size
    previous_at: datetime | None = None
    for index, sample in enumerate(values):
        state = provisional._advance(state, sample, previous_at)
        previous_at = sample.observed_at
        if index == len(values) - 1:
            continue
        rows.append(provisional._readout_row(state, sample))
        next_sample = values[index + 1]
        mean_targets.append(
            (math.log1p(next_sample.mean_vtec_tecu) - mean_center) / mean_scale
        )
        median_targets.append(
            (math.log1p(next_sample.median_vtec_tecu) - median_center)
            / median_scale
        )

    mean_readout = _ridge_fit(
        tuple(rows[index] for index in fit_pair_indices),
        tuple(mean_targets[index] for index in fit_pair_indices),
        ridge_penalty,
    )
    median_readout = _ridge_fit(
        tuple(rows[index] for index in fit_pair_indices),
        tuple(median_targets[index] for index in fit_pair_indices),
        ridge_penalty,
    )

    calibrated = _replace_readout_and_anchor(
        provisional,
        mean_readout=mean_readout,
        median_readout=median_readout,
        anchor_state=state,
    )
    mean_residuals: list[float] = []
    median_residuals: list[float] = []
    for index in calibration_pair_indices:
        predicted_mean = calibrated._decode_target(
            _dot(mean_readout, rows[index]), metric_index=0
        )
        predicted_median = calibrated._decode_target(
            _dot(median_readout, rows[index]), metric_index=1
        )
        target = values[index + 1]
        mean_residuals.append(target.mean_vtec_tecu - predicted_mean)
        median_residuals.append(target.median_vtec_tecu - predicted_median)
    mean_residual_center, mean_residual_scale = _robust_residual_calibration(
        mean_residuals,
        floor=residual_scale_floor_tecu,
    )
    median_residual_center, median_residual_scale = _robust_residual_calibration(
        median_residuals,
        floor=residual_scale_floor_tecu,
    )
    return _replace_calibration(
        calibrated,
        mean_center=mean_residual_center,
        mean_scale=mean_residual_scale,
        median_center=median_residual_center,
        median_scale=median_residual_scale,
    )


def _replace_readout_and_anchor(
    model: MambaModel,
    *,
    mean_readout: tuple[float, ...],
    median_readout: tuple[float, ...],
    anchor_state: tuple[float, ...],
) -> MambaModel:
    payload = _model_arguments(model)
    payload.update(
        mean_readout=mean_readout,
        median_readout=median_readout,
        anchor_state=anchor_state,
    )
    return MambaModel(**payload)


def _replace_calibration(
    model: MambaModel,
    *,
    mean_center: float,
    mean_scale: float,
    median_center: float,
    median_scale: float,
) -> MambaModel:
    payload = _model_arguments(model)
    payload.update(
        mean_residual_center_tecu=mean_center,
        mean_residual_scale_tecu=mean_scale,
        median_residual_center_tecu=median_center,
        median_residual_scale_tecu=median_scale,
    )
    return MambaModel(**payload)


def _model_arguments(model: MambaModel) -> dict[str, Any]:
    return {
        field.name: getattr(model, field.name)
        for field in fields(model)
    }


def _reference_recurrence(
    state_size: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
]:
    denominator = max(1, state_size - 1)
    decay_rates = tuple(
        math.exp(math.log(0.02) + index / denominator * math.log(64.0))
        for index in range(state_size)
    )
    delta_biases = tuple(
        -0.35 + 0.7 * index / denominator for index in range(state_size)
    )
    input_biases = tuple(
        _fixed_weight("input-bias", index, 0, 0.2)
        for index in range(state_size)
    )
    output_biases = tuple(
        _fixed_weight("output-bias", index, 0, 0.2)
        for index in range(state_size)
    )
    return (
        decay_rates,
        delta_biases,
        _fixed_matrix("delta", state_size, scale=0.35),
        input_biases,
        _fixed_matrix("input", state_size, scale=0.9),
        output_biases,
        _fixed_matrix("output", state_size, scale=0.45),
    )


def _fixed_matrix(
    family: str,
    rows: int,
    *,
    scale: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            _fixed_weight(family, row, column, scale / math.sqrt(_FEATURE_COUNT))
            for column in range(_FEATURE_COUNT)
        )
        for row in range(rows)
    )


def _fixed_weight(family: str, row: int, column: int, scale: float) -> float:
    digest = sha256(
        f"{ALGORITHM}\x1f{family}\x1f{row}\x1f{column}".encode("ascii")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return (unit * 2.0 - 1.0) * scale


def _ridge_fit(
    rows: Sequence[tuple[float, ...]],
    targets: Sequence[float],
    penalty: float,
) -> tuple[float, ...]:
    if not rows or len(rows) != len(targets):
        raise MambaModelError("ridge readout needs matching non-empty rows and targets")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise MambaModelError("ridge readout rows have inconsistent widths")
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    right = [0.0 for _ in range(width)]
    for left_index in range(width):
        right[left_index] = math.fsum(
            row[left_index] * target for row, target in zip(rows, targets)
        )
        for right_index in range(left_index, width):
            value = math.fsum(
                row[left_index] * row[right_index] for row in rows
            )
            if left_index == right_index and left_index != 0:
                value += penalty
            gram[left_index][right_index] = value
            gram[right_index][left_index] = value
    return _solve(gram, right)


def _solve(matrix: list[list[float]], right: list[float]) -> tuple[float, ...]:
    size = len(right)
    augmented = [matrix[index][:] + [right[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise MambaModelError("ridge readout system is numerically singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    solution = tuple(augmented[index][-1] for index in range(size))
    if not all(math.isfinite(value) for value in solution):
        raise MambaModelError("ridge readout produced non-finite coefficients")
    return solution


def _robust_log_scaling(values: Iterable[float]) -> tuple[float, float]:
    transformed = tuple(math.log1p(value) for value in values)
    center = median(transformed)
    absolute_deviations = tuple(abs(value - center) for value in transformed)
    # 1.4826 makes MAD comparable to standard deviation for normal residuals.
    return center, max(0.01, 1.4826 * median(absolute_deviations))


def _robust_residual_calibration(
    residuals: Sequence[float],
    *,
    floor: float,
) -> tuple[float, float]:
    if len(residuals) < 3 or not all(math.isfinite(value) for value in residuals):
        raise MambaModelError("robust calibration needs at least three finite residuals")
    center = median(residuals)
    mad_scale = 1.4826 * median(abs(value - center) for value in residuals)
    return center, max(float(floor), mad_scale)


def _ordered_samples(
    samples: Iterable[TrainingSample],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[TrainingSample, ...]:
    values = tuple(samples)
    if not values and not allow_empty:
        raise MambaModelError(f"{name} must not be empty")
    if any(not isinstance(sample, TrainingSample) for sample in values):
        raise TypeError(f"{name} must contain TrainingSample values")
    for previous, current in zip(values, values[1:]):
        if current.observed_at <= previous.observed_at:
            raise MambaModelError(f"{name} must be strictly chronological")
    return values


def _median_positive_deltas(samples: Sequence[TrainingSample]) -> float:
    deltas = tuple(
        (current.observed_at - previous.observed_at).total_seconds()
        for previous, current in zip(samples, samples[1:])
    )
    if not deltas or any(value <= 0.0 for value in deltas):
        raise MambaModelError("training sample cadence must be positive")
    cadence = float(median(deltas))
    tolerance = max(1.0, cadence * 0.05)
    if any(abs(value - cadence) > tolerance for value in deltas):
        raise MambaModelError(
            "training samples must use one regular cadence; represent gaps explicitly"
        )
    return cadence


def _training_data_hash(samples: Sequence[TrainingSample]) -> str:
    return sha256(
        _canonical_json({"samples": [sample.to_dict() for sample in samples]})
    ).hexdigest()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise MambaModelError("internal vector dimensions do not agree")
    return math.fsum(first * second for first, second in zip(left, right))


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _datetime_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_datetime(value: Any) -> datetime:
    text = _text(value, "datetime")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise MambaArtifactError("model artifact datetime is invalid") from error
    _aware(parsed, "model artifact datetime")
    return parsed.astimezone(UTC)


def _sample_from_payload(payload: dict[str, Any]) -> TrainingSample:
    try:
        return TrainingSample(
            observed_at=_parse_datetime(payload["observed_at"]),
            mean_vtec_tecu=_number(payload["mean_vtec_tecu"], "sample mean"),
            median_vtec_tecu=_number(
                payload["median_vtec_tecu"], "sample median"
            ),
            coverage_fraction=_number(
                payload["coverage_fraction"], "sample coverage"
            ),
        )
    except KeyError as error:
        raise MambaArtifactError(
            f"model anchor sample is missing field {error.args[0]!r}"
        ) from error


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise MambaArtifactError(f"model artifact {key!r} must be an object")
    return value


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise MambaArtifactError(f"model {name} must be an array")
    return tuple(_number(item, name) for item in value)


def _float_matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list):
        raise MambaArtifactError(f"model {name} must be an array")
    return tuple(_float_tuple(row, name) for row in value)


def _number(value: Any, name: str) -> float:
    _finite_number(value, name)
    return float(value)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MambaArtifactError(f"model {name} must be an integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MambaArtifactError(f"model {name} must be a non-empty string")
    return value


def _finite_number(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MambaModelError(f"{name} must be a finite number")


def _aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MambaModelError(f"{name} must be timezone-aware")


def _validate_digest(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MambaArtifactError(f"model {name} must be 64 lowercase hex characters")


def _validate_vector(
    value: Any,
    size: int,
    name: str,
    *,
    positive: bool = False,
) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise MambaArtifactError(f"model {name} must contain {size} values")
    for item in value:
        _finite_number(item, f"model {name}")
        if positive and item <= 0.0:
            raise MambaArtifactError(f"model {name} values must be positive")


__all__ = [
    "ALGORITHM",
    "ARTIFACT_FORMAT",
    "FEATURE_NAMES",
    "MambaArtifactError",
    "MambaModel",
    "MambaModelError",
    "ScoreResult",
    "TrainingSample",
    "fit_model",
]
