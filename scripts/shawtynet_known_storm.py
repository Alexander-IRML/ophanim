"""Reproduce a bounded, retrospective known-storm sanity check using real CODE GIM.

Downloads only three final IONEX files into a separate validation directory.
It does not ingest into OPHANIM's existing archive/database. The preceding day
is a descriptive same-UT comparison, NOT an independently verified quiet day.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path


DATES = (date(2024, 5, 9), date(2024, 5, 10), date(2024, 5, 11))
BOUNDS = {"south": 20.0, "north": 42.0, "west": -112.0, "east": -88.0}
REFERENCES = {
    "noaa_g5": "https://www.swpc.noaa.gov/news/g5-conditions-observed",
    "usgs_storm": "https://www.usgs.gov/programs/geomagnetism/science/may-10-2024-magnetic-disturbance",
}


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch_sources(root):
    from ophanim.gim import CodeGIMSource, GIMEdition, code_gim_candidate

    directory = root / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for day in DATES:
        candidate = code_gim_candidate(GIMEdition.FINAL, day)
        path = directory / candidate.filename
        receipt = path.with_suffix(path.suffix + ".json")
        if receipt.exists() and path.exists():
            record = json.loads(receipt.read_text())
            if sha256(path.read_bytes()).hexdigest() != record["checksum_sha256"]:
                raise ValueError(f"Cached source checksum changed: {path}")
        elif receipt.exists() or path.exists():
            raise ValueError(f"Incomplete source cache; choose a new output directory: {path}")
        else:
            fetched = CodeGIMSource().fetch_date(day, edition=GIMEdition.FINAL)
            payload = fetched.loaded_source.content
            # 'xb' refuses to replace any existing artifact, even during a race.
            with path.open("xb") as handle:
                handle.write(payload)
            record = {
                "filename": candidate.filename, "product_date": day.isoformat(),
                "edition": "final", "checksum_sha256": sha256(payload).hexdigest(),
                "bytes": len(payload), "source_uri": fetched.source_uri,
                "resolved_uri": fetched.resolved_uri,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "available_at_definition": "local acquisition, NOT historical publication",
            }
            _json(receipt, record)
        records.append(record)
        print(json.dumps({"source": record}), flush=True)
    return records


def read_sources(root, records):
    import numpy as np
    import xarray as xr
    from ophanim.domain import SourceArtifact
    from ophanim.ionex import IONEXV1Parser
    from ophanim.sensing import read_ionex_sequence
    from ophanim.sensing.sequence import _finish_dataset

    daily, provenance = [], []
    for record in records:
        path = root / "sources" / record["filename"]
        artifact = SourceArtifact(
            artifact_id="known-storm-" + record["checksum_sha256"],
            provider="code", product="gim", revision="final", revision_priority=20,
            parser_version=IONEXV1Parser.parser_version,
            checksum_sha256=record["checksum_sha256"], storage_ref=str(path),
            ingested_at=datetime.fromisoformat(record["downloaded_at"]),
            source_uri=record["source_uri"],
        )
        native = read_ionex_sequence(path, artifact=artifact, bounds=BOUNDS)
        provenance.extend(native.attrs["source_metadata"]["revision_history"])
        daily.append(native)
    combined = xr.concat(daily, dim="time", combine_attrs="drop")
    # Adjacent IONEX products overlap at midnight. Later product date wins;
    # this explicit choice and every selected source ID remain in provenance.
    _, reverse_indices = np.unique(combined.time.values[::-1], return_index=True)
    indices = np.sort(len(combined.time) - 1 - reverse_indices)
    combined = combined.isel(time=indices).sortby("time")
    combined = _finish_dataset(combined, provenance, {}, None, None, None, BOUNDS)
    combined.attrs["overlap_policy"] = "later daily final product wins at duplicate midnight"
    return combined


def _finite(value):
    import numpy as np
    return float(value) if np.isfinite(value) else None


def summarize(root, native, science):
    import numpy as np
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    event = json.loads((science / "event.json").read_text())
    packet = json.loads((science / "science_packet.json").read_text())
    dynamic = xr.open_zarr(science / "dynamic.zarr", consolidated=True).load()
    weights = np.cos(np.deg2rad(native.lat))
    mean = native.tec.weighted(weights).mean(("lat", "lon")).values
    times = native.time.values
    comparison = np.full(len(times), np.nan)
    reference = {str(t.astype("datetime64[h]")): v for t, v in zip(times[:24], mean[:24])}
    for i, time in enumerate(times):
        hour = time.astype("datetime64[h]").astype(int) % 24
        reference_time = np.datetime64("2024-05-09T00", "h") + np.timedelta64(int(hour), "h")
        comparison[i] = reference.get(str(reference_time), np.nan)
    mean_delta = mean - comparison
    finite_count = np.isfinite(dynamic.dtec.values).sum(axis=(1, 2))
    dtec_p95 = np.array([
        np.nanpercentile(np.abs(frame), 95) if count else np.nan
        for frame, count in zip(dynamic.dtec.values, finite_count)
    ])
    start = np.datetime64("2024-05-10T16:06")
    g5 = np.datetime64("2024-05-10T22:54")
    storm_mask = (times >= start) & (times < np.datetime64("2024-05-12"))
    peak_index = int(np.flatnonzero(storm_mask)[np.argmax(mean_delta[storm_mask])])
    before_mask = (times >= np.datetime64("2024-05-09T18")) & (times < np.datetime64("2024-05-10"))
    after_mask = (times >= np.datetime64("2024-05-10T18")) & (times < np.datetime64("2024-05-11"))
    summary = {
        "purpose": "qualitative known-event sanity check, not causal attribution or detection calibration",
        "source_kind": "actual CODE final IONEX, not synthetic",
        "independent_event_references": REFERENCES,
        "independent_storm_commencement_utc": str(start) + "Z",
        "independent_first_g5_utc": str(g5) + "Z",
        "analysis_mode": "retrospective",
        "comparison": "same UTC hour on May 9; not independently verified quiet control",
        "bounds": BOUNDS, "native_shape": dict(native.sizes),
        "science_shape": dict(dynamic.sizes),
        "source_metadata": native.attrs["source_metadata"],
        "science_run": str(science), "diagnostics": str(science / "diagnostics" / "report.html"),
        "event_display_class": event["display_class"],
        "event_channel_status": event["channel_status"],
        "event_confidence": event["event_confidence"],
        "capabilities": packet.get("capabilities"),
        "maximum_storm_interval_regional_same_ut_increase": {
            "time": str(times[peak_index]) + "Z", "tecu": float(mean_delta[peak_index]),
            "native_regional_mean_tecu": float(mean[peak_index]),
            "may_9_same_ut_regional_mean_tecu": float(comparison[peak_index]),
        },
        "regional_mean_tecu_18_to_23_utc": {
            "may_9": float(np.mean(mean[before_mask])), "may_10": float(np.mean(mean[after_mask])),
        },
        "regional_p95_absolute_12h_median_residual_tecu_18_to_23_utc": {
            "may_9": float(np.nanmean(dtec_p95[before_mask])),
            "may_10": float(np.nanmean(dtec_p95[after_mask])),
        },
        "native_regional_timeseries": [
            {"time": str(t) + "Z", "area_weighted_mean_tecu": float(v),
             "difference_from_may_9_same_ut_tecu": float(delta),
             "science_p95_absolute_12h_median_residual_tecu": _finite(residual)}
            for t, v, delta, residual in zip(times, mean, mean_delta, dtec_p95)
        ],
        "limits": [
            "CODE is a smoothed estimated global source product, not direct instrument samples at each node.",
            "75 km analysis interpolation adds no resolving power to native 5 by 2.5 degree cells.",
            "12-hour median residual contains diurnal structure and is not a quiet-day storm anomaly.",
            "Same-UT May 9 reference is one comparison day, not a validated climatology or quiet control.",
            "Temporal coincidence does not identify a physical cause or validate flow speed/direction.",
            "Hourly sampling cannot support this configured 30–180 minute wave search; withholding is expected.",
        ],
    }
    _json(root / "summary.json", summary)
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(times, mean, label="Native CODE regional TEC")
    axes[0].plot(times, comparison, "--", label="May 9 same-UT comparison")
    axes[0].set_ylabel("Regional mean [TECU]")
    axes[0].legend(loc="upper left")
    axes[1].plot(times, mean_delta, color="tab:purple")
    axes[1].axhline(0, color="0.6", lw=0.7)
    axes[1].set_ylabel("Difference from May 9\nsame UTC hour [TECU]")
    axes[2].plot(dynamic.time, dtec_p95, color="tab:orange")
    axes[2].set_ylabel("P95 |12-hour median\nresidual| [TECU]")
    for axis in axes:
        axis.axvline(start, color="0.4", ls=":", label="USGS commencement")
        axis.axvline(g5, color="tab:red", ls=":", label="NOAA first G5")
        axis.grid(alpha=0.2)
    axes[2].legend(loc="upper right")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M UTC"))
    figure.suptitle("Actual CODE GIM, broad Texas region · qualitative known-event check\n"
                    "Retrospective; comparison day is NOT an independently verified quiet control")
    figure.savefig(root / "known-storm-timeseries.png", dpi=145)
    plt.close(figure)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"source_metadata", "native_regional_timeseries", "capabilities"}}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("var/shawtynet-validation/known-storm"))
    parser.add_argument("--fetch-only", action="store_true")
    arguments = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    root = arguments.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = fetch_sources(root)
    if arguments.fetch_only:
        return 0
    from ophanim.dynamics import AnalysisConfig
    from ophanim.science_runs import analyze_run
    native = read_sources(root, records)
    print(json.dumps({"native_sizes": dict(native.sizes), "start": str(native.time.values[0]),
                      "end": str(native.time.values[-1]),
                      "native_cadence_seconds": native.attrs["source_metadata"]["cadence_seconds"]}), flush=True)
    config = AnalysisConfig(
        science_spacing_km=75.0, smooth_sigma_km=100.0, structure_sigma_km=200.0,
        short_window_minutes=360.0, baseline_method="rolling_median",
        baseline_window_minutes=720.0, wave_window_hours=24.0,
        target_time="2024-05-10T23:00:00Z", max_cube_cells=200_000,
    )
    science = analyze_run(native, config, root / "science", request={
        "validation": "real-known-storm", "source_records": records,
        "bounds": BOUNDS, "independent_references": REFERENCES,
    })
    summarize(root, native, science)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
