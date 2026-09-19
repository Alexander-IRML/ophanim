"""Composable source → science pipeline; no art or persistence dependencies."""

from __future__ import annotations

import logging

import numpy as np

from .capabilities import assess_capabilities
from .coords import project_dataset
from .derivatives import add_derivatives
from .events import score_events
from .flow import add_flow
from .preprocessing import preprocess
from .regularize import eligible_sequence, iso_time, regularize_time, utc64
from .schemas import AnalysisConfig, AnalysisError, AnalysisResult
from .waves import estimate_wave

LOGGER = logging.getLogger(__name__)


def prepare_dataset(raw, config):
    """Prepare reusable retrospective operators once; causal callers pass a cutoff."""
    source = eligible_sequence(raw, config)
    if config.target_time and utc64(config.target_time) < source.time.values[0]:
        raise AnalysisError("target_time is before available observations")
    LOGGER.info("Dynamics source: %s to %s, %s", source.time.values[0], source.time.values[-1], dict(source.sizes))
    projected = project_dataset(source, config)
    capabilities = assess_capabilities(projected, config)
    regular = regularize_time(projected, config)
    if not np.isfinite(regular.tec.values).any():
        raise AnalysisError("analysis region contains no usable observations")
    data, baseline = preprocess(regular, config)
    LOGGER.info("Dynamics preprocessing: cadence=%s, missing=%.3f, baseline=%s", data.attrs.get("cadence_seconds"), float(data.tec.isnull().mean()), baseline)
    data = add_derivatives(data, config)
    return data, baseline, capabilities, source


def finish_dataset(data, event, capabilities, baseline, source, config):
    """Attach the single-target compatibility metadata without performing inference."""
    wave = event["wave"]
    data["wave_confidence"] = ((), float(wave.get("confidence", 0)), {"units": "1", "semantic_class": "inferred", "semantic": "regional heuristic wave confidence, not probability"})
    data["event_confidence"] = ((), event["event_confidence"], {"units": "1", "semantic_class": "inferred", "semantic": "regional event interpretation heuristic"})
    data.attrs.update(schema_version="ophanim.dynamic/1", analysis_config=config.to_dict(), capabilities=capabilities, baseline_status=baseline, analysis_time=event["time"], target_time=config.target_time or event["time"], analysis_mode=config.analysis_mode, as_of=config.as_of or "", observed_range=[iso_time(source.time.values[0]), iso_time(source.time.values[-1])], science_semantics="TEC-product dynamics; apparent image-feature motion is not plasma velocity; no artistic geometry in this dataset", software_algorithm_version="dynamics/1")
    return data


def analyze_dataset(raw, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Produce a frozen-input, source-aware, confidence-separated dynamic state.

    Partial scientific channels are represented with NaN physical values, zero
    support scores and explicit statuses. Invalid coordinates/configuration fail
    rather than manufacturing an apparently quiet zero-filled result.
    """
    config = config or AnalysisConfig()
    data, baseline, capabilities, source = prepare_dataset(raw, config)
    data, flow_records = add_flow(data, config, capabilities)
    wave = estimate_wave(data, config, capabilities)
    event = score_events(data, wave, config, baseline, flow_records)
    data = finish_dataset(data, event, capabilities, baseline, source, config)
    LOGGER.info("Dynamics wave=%s; event=%s scores=%s", wave["status"], event["display_class"], event["event_scores"])
    return AnalysisResult(data, event, capabilities)
