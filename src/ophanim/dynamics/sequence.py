"""Bounded, independently gated scientific interpretations at exact epochs.

Retrospective preprocessing and derivatives are reused. Only requested adjacent
flow pairs and wave windows are analyzed. Causal requests deliberately recompute
each prefix so future normalization, revisions and derivatives cannot leak in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

from .schemas import AnalysisConfig, AnalysisError


@dataclass(frozen=True, slots=True)
class SequenceBudget:
    max_frames: int = 12
    max_cube_cells: int = 2_000_000
    max_work_cells: int = 12_000_000

    def __post_init__(self):
        limits = {"max_frames": 24, "max_cube_cells": 2_000_000, "max_work_cells": 24_000_000}
        for name, limit in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
                raise AnalysisError(f"sequence {name} must be an integer from 1 through {limit}")


@dataclass(frozen=True, slots=True)
class SequenceAnalysisResult:
    dataset: Any
    event: dict[str, Any]
    capabilities: dict[str, Any]
    timeline: dict[str, Any]


def normalize_frame_times(frame_times, *, budget=None):
    """Canonicalize explicit UTC epochs without silently dropping duplicate frames."""
    import numpy as np
    from .regularize import utc64, iso_time

    budget = budget or SequenceBudget()
    if not isinstance(budget, SequenceBudget):
        raise AnalysisError("sequence_budget must be a SequenceBudget")
    if not isinstance(frame_times, (list, tuple)) or not 1 <= len(frame_times) <= budget.max_frames:
        raise AnalysisError(f"frame_times must contain 1 through {budget.max_frames} exact UTC epochs")
    times = []
    for value in frame_times:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise AnalysisError("frame time must be an explicit UTC timestamp") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise AnalysisError("frame time must include a timezone")
        elif not isinstance(value, np.datetime64):
            raise AnalysisError("frame time must be a timezone-aware ISO string or UTC datetime64")
        timestamp = utc64(value)
        if np.isnat(timestamp):
            raise AnalysisError("frame time cannot be NaT")
        times.append(timestamp)
    if len(set(int(value.astype('int64')) for value in times)) != len(times):
        raise AnalysisError("frame_times contains duplicate epochs")
    return [iso_time(value) for value in sorted(times)]


def _frame(event, capabilities, config):
    import numpy as np
    from .regularize import utc64, iso_time

    target = utc64(event["time"])
    statuses = event.get("channel_status", {})
    baseline = statuses.get("baseline", {})
    anomaly_available = baseline.get("status") in {"estimated", "available", "supported"}
    # Event metrics preserve explicit valid anomaly support even if a baseline
    # backend uses another descriptive status for its successful estimate.
    anomaly_available = anomaly_available or event.get("metrics", {}).get("anomaly_support", 0) > 0
    wave_status = event.get("wave", {}).get("status")
    flow_status = statuses.get("flow", {}).get("status")
    status = "analyzed" if wave_status == "estimated" and flow_status == "estimated" else "partial"
    if not anomaly_available:
        status = "unavailable"
    return {
        "time": event["time"], "status": status, "event": event,
        "analysis_window": {
            "analysis_mode": config.analysis_mode, "as_of": config.as_of,
            "wave_start": iso_time(target - np.timedelta64(round(config.wave_window_hours * 3600 * 1e9), "ns")),
            "wave_end": event["time"],
            "wave_observed_duration_seconds": capabilities.get("wave_observed_duration_seconds"),
            "wave_observation_count": capabilities.get("wave_observation_count"),
            "admissible_period_seconds": capabilities.get("admissible_period_seconds"),
            "flow": "adjacent source-supported pair ending at this exact epoch",
            "baseline": "shared retrospective preprocessing" if config.analysis_mode == "retrospective" else "independent causal prefix",
        },
        "capabilities": capabilities,
    }


def analyze_sequence(raw, config=None, frame_times=None, *, budget=None, cancelled=None):
    """Return exact-time events without repeating the whole retrospective cube.

    ``config.target_time`` remains the scientific report target, independently
    of an artistic display frame. It is automatically included in the bounded
    requested set. The caller must reserve a frame if its chosen target differs.
    Unscheduled epochs never acquire copied event or wave confidence.
    """
    import numpy as np
    import xarray as xr
    from ophanim.core.state import TIMELINE_SCHEMA, validate_timeline
    from .capabilities import assess_capabilities
    from .events import score_events
    from .flow import add_flow
    from .pipeline import analyze_dataset, prepare_dataset, finish_dataset
    from .regularize import eligible_sequence, utc64, iso_time
    from .waves import estimate_wave

    config = config or AnalysisConfig()
    budget = budget or SequenceBudget()
    times = normalize_frame_times(frame_times, budget=budget)

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("Scientific sequence cancelled between inference windows")

    check()
    source = eligible_sequence(raw, config)
    target = utc64(config.target_time) if config.target_time else source.time.values[-1]
    prior = source.time.values[source.time.values <= target]
    if not len(prior):
        raise AnalysisError("sequence scientific target precedes source observations")
    target = prior[-1]
    target_text = iso_time(target)
    if target_text not in times:
        times = normalize_frame_times([*times, target_text], budget=budget)
    if any(utc64(value) not in source.time.values for value in times):
        raise AnalysisError("sequence frame must match an exact eligible source epoch; interpolation is not an observation")
    bounded_config = replace(config, max_cube_cells=min(config.max_cube_cells, budget.max_cube_cells))
    frames, selected_results = [], []
    work_cells = 0
    if config.analysis_mode == "causal":
        # Never reuse final-prefix normalization/derivatives for earlier times.
        for timestamp in times:
            check()
            window_config = replace(bounded_config, target_time=timestamp)
            prefix = eligible_sequence(raw, window_config)
            source_cells = int(prefix.tec.size)
            work_cells += source_cells
            if work_cells > budget.max_work_cells:
                raise AnalysisError("causal sequence exceeds max_work_cells before another analysis")
            result = analyze_dataset(prefix, window_config)
            if result.dataset.tec.size > source_cells:
                work_cells += result.dataset.tec.size - source_cells
                if work_cells > budget.max_work_cells:
                    raise AnalysisError("projected causal sequence exceeds max_work_cells")
            index = np.flatnonzero(result.dataset.time.values == utc64(timestamp))
            if not len(index):
                raise AnalysisError("requested source epoch is not on the configured regular science grid")
            selected_results.append(result.dataset.isel(time=[int(index[0])]))
            frames.append(_frame(result.event, result.capabilities, window_config))
        data = xr.concat(selected_results, dim="time", data_vars="all", coords="minimal", compat="override")
        chosen = next(frame for frame in frames if frame["time"] == target_text)
        data.attrs.update(selected_results[-1].attrs)
        data.attrs.update(analysis_config=config.to_dict(), analysis_time=target_text, target_time=target_text)
    else:
        check()
        data, baseline, _, prepared_source = prepare_dataset(source, bounded_config)
        if any(utc64(value) not in data.time.values for value in times):
            raise AnalysisError("requested source epoch is not on the configured regular science grid")
        work_cells = int(data.tec.size)
        for timestamp in times:
            end = utc64(timestamp)
            start = end - np.timedelta64(round(config.wave_window_hours * 3600 * 1e9), "ns")
            work_cells += int(np.sum((data.time.values >= start) & (data.time.values <= end))) * data.sizes["x"] * data.sizes["y"]
        if work_cells > budget.max_work_cells:
            raise AnalysisError("sequence preprocessing/wave windows exceed max_work_cells")
        flow_capabilities = assess_capabilities(data, bounded_config)
        data, records = add_flow(data, bounded_config, flow_capabilities, frame_times=times)
        for timestamp in times:
            check()
            window_config = replace(bounded_config, target_time=timestamp)
            capabilities = assess_capabilities(data, window_config)
            wave = estimate_wave(data, window_config, capabilities)
            matching = [record for record in records if utc64(record["time"]) == utc64(timestamp)]
            event = score_events(data, wave, window_config, baseline, matching)
            frames.append(_frame(event, capabilities, window_config))
        chosen = next(frame for frame in frames if frame["time"] == target_text)
        data = finish_dataset(data, chosen["event"], chosen["capabilities"], baseline, prepared_source, config)
    check()
    for field, getter in (("wave_confidence", lambda event: event["wave"].get("confidence", 0)),
                          ("event_confidence", lambda event: event["event_confidence"])):
        values = np.full(data.sizes["time"], np.nan)
        for frame in frames:
            values[data.time.values == utc64(frame["time"])] = getter(frame["event"])
        data[field] = (("time",), values, {"units": "1", "semantic_class": "inferred",
                       "semantic": "exact-epoch heuristic inference support, not probability; NaN means unscheduled"})
    scheduled = np.asarray([any(value == utc64(frame["time"]) for frame in frames) for value in data.time.values])
    data["inference_scheduled"] = (("time",), scheduled, {"units": "1", "semantic_class": "derived",
                                     "semantic": "explicitly requested inference epoch; false is not an inferred quiet state"})
    timeline = {"schema_version": TIMELINE_SCHEMA, "analysis_mode": config.analysis_mode,
                "scientific_target_time": target_text, "frames": frames,
                "budget": asdict(budget), "work_cells": work_cells,
                "semantics": "Independent exact-epoch wave/event windows; no event or phase extrapolation beyond these epochs"}
    validate_timeline(timeline, data.time.values)
    data.attrs.update(event_timeline_schema=TIMELINE_SCHEMA, inference_frame_times=times,
                      sequence_budget=asdict(budget), inference_work_cells=work_cells)
    return SequenceAnalysisResult(data, chosen["event"], chosen["capabilities"], timeline)


__all__ = ["SequenceBudget", "SequenceAnalysisResult", "normalize_frame_times", "analyze_sequence"]
