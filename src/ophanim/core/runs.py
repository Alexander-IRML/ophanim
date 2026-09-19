"""Scientific run publication; no experimental or artistic dependencies."""
from __future__ import annotations

from hashlib import sha256
import logging
from pathlib import Path
from typing import Any

from .artifacts import (RUN_SCHEMA, canonical_json, dataset_sha256, json_value,
                        software_identity, write_json, _publish, _write_dataset)

LOGGER = logging.getLogger(__name__)

def analyze_run(raw: Any, config: Any, output_root: str | Path, *,
                request: dict[str, Any] | None = None, frame_times=None,
                sequence_budget=None, cancelled=None) -> Path:
    """Publish a scientific snapshot, optionally with bounded exact-time inference.

    ``frame_times`` requests independently analyzed event/wave windows, not a
    change to the report's scientific target or an artistic preview selection.
    """
    from ophanim.dynamics import analyze_dataset
    from ophanim.science_reports import science_diagnostics
    from .state import EngineeredStateAdapter, STATE_SCHEMA

    LOGGER.info("Analysis input dimensions=%s mode=%s", dict(raw.sizes), config.analysis_mode)
    resolved = json_value(config)
    software = software_identity()
    identity = {"input_sha256": dataset_sha256(raw), "config": resolved,
                "software": software, "schema": RUN_SCHEMA}
    if frame_times is not None:
        from ophanim.dynamics.sequence import SequenceBudget, normalize_frame_times
        sequence_budget = sequence_budget or SequenceBudget()
        frame_times = normalize_frame_times(frame_times, budget=sequence_budget)
        identity["sequence"] = {"frame_times": frame_times, "budget": json_value(sequence_budget)}
    elif sequence_budget is not None:
        raise ValueError("sequence_budget requires explicit frame_times")
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage: Path) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Scientific publication cancelled before analysis")
        timeline = None
        if frame_times is None:
            result = analyze_dataset(raw, config)
        else:
            from ophanim.dynamics.sequence import analyze_sequence
            result = analyze_sequence(raw, config, frame_times, budget=sequence_budget, cancelled=cancelled)
            timeline = result.timeline
        state = EngineeredStateAdapter().from_result(result, timeline=timeline)
        if cancelled is not None and cancelled():
            raise InterruptedError("Scientific publication cancelled before artifact writes")
        result.dataset.attrs["run_id"] = run_id
        result.dataset.attrs["science_run_id"] = run_id
        _write_dataset(result.dataset, stage / "dynamic.zarr")
        write_json(stage / "config.json", resolved)
        if request is not None:
            write_json(stage / "request.json", request)
        write_json(stage / "event.json", result.event)
        if timeline is not None:
            write_json(stage / "events.json", timeline)
        write_json(stage / "state_contract.json", {
            "schema_version": STATE_SCHEMA, "source_kind": state.dataset.attrs["source_kind"],
            "dataset": "dynamic.zarr", "producer": state.provenance,
            "fields": {name: {"dimensions": list(value.dims), **value.attrs}
                       for name, value in state.dataset.data_vars.items()},
            "event": "event.json", "timeline": "events.json" if timeline is not None else None,
        })
        source = raw.attrs.get("source_metadata", {})
        source_kind = state.dataset.attrs["source_kind"]
        synthetic = source_kind == "synthetic"
        packet = {
            "schema": "ophanim-dynamic-state/1", "run_id": run_id,
            "input_sha256": identity["input_sha256"],
            "dynamic_sha256": dataset_sha256(result.dataset),
            "source": source, "source_kind": source_kind,
            "state_contract": "state_contract.json", "state_schema": STATE_SCHEMA,
            "event_timeline": "events.json" if timeline is not None else None,
            "config": resolved, "software": software,
            "capabilities": result.capabilities, "event": result.event,
            "analysis_mode": config.analysis_mode,
            "as_of": config.as_of,
            "time_start": str(result.dataset.time.values[0]),
            "time_end": str(result.dataset.time.values[-1]),
            "dimensions": dict(result.dataset.sizes),
            "metadata": result.dataset.attrs,
            "fields": {name: {"dimensions": list(value.dims), **value.attrs}
                       for name, value in result.dataset.data_vars.items()},
            "semantics": {
                "measured": [] if synthetic else ["source TEC product, itself an estimate from GNSS observations",
                             "source coordinates/timestamps", "source-provided RMS if available"],
                "synthetic": ["chosen mathematical TEC fixture", "generated sample coordinates/timestamps",
                              "chosen synthetic noise and support"] if synthetic else [],
                "derived": ["background", "dTEC", "derivatives", "structure tensor"],
                "inferred": ["apparent TEC-feature motion", "wave interpretation",
                             "uncalibrated event/confidence scores"],
                "artistic": [],
            },
            "limitations": ["TEC is column integrated, not a 3-D electron-density field",
                            "Motion is not plasma velocity; low residual is not proof of transport",
                            "Source presence and source accuracy are different quantities",
                            "Interpolation adds no observations or guaranteed physical resolution"],
        }
        write_json(stage / "science_packet.json", packet)
        science_diagnostics(result.dataset, result.event, packet, stage / "diagnostics")
        if cancelled is not None and cancelled():
            raise InterruptedError("Scientific publication cancelled before completion")

    return _publish(Path(output_root), run_id, "science", identity, build)
