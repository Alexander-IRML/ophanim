"""Inspectable, headless diagnostics; no scientific inference is performed here."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _figure():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _field_plot(dataset: Any, name: str, destination: Path, *, frame_index: int = -1) -> None:
    import numpy as np

    figure, axes = _figure()
    if name not in dataset:
        axes.text(0.5, 0.5, "Channel unavailable", ha="center", va="center")
    else:
        field = dataset[name]
        if "time" in field.dims:
            field = field.isel(time=frame_index)
        values = np.asarray(field.values, dtype=float)
        if values.ndim == 2 and np.isfinite(values).any():
            extent = None
            if "x" in field.coords and "y" in field.coords:
                extent = [float(field.x[0]) / 1000, float(field.x[-1]) / 1000,
                          float(field.y[0]) / 1000, float(field.y[-1]) / 1000]
            artist = axes.imshow(values, origin="lower", aspect="auto", extent=extent,
                                 cmap="coolwarm" if "dtec" in name else "viridis")
            figure.colorbar(artist, ax=axes, label=field.attrs.get("units", ""))
            axes.set_xlabel("Projection x (km)")
            axes.set_ylabel("Projection y (km)")
        else:
            axes.text(0.5, 0.5, "Unavailable / unsupported — not zero", ha="center", va="center")
    axes.set_title(name.replace("_", " "))
    figure.savefig(destination, dpi=110)
    figure.clear()


def _report(destination: Path, title: str, images: list[str], metadata: dict[str, Any]) -> None:
    from ophanim.core.artifacts import json_value

    cards = "\n".join(f'<figure><img src="{html.escape(name, quote=True)}" '
                      f'alt="{html.escape(Path(name).stem)}"><figcaption>'
                      f'{html.escape(Path(name).stem)}</figcaption></figure>' for name in images)
    source = html.escape(json.dumps(json_value(metadata), indent=2, sort_keys=True))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font:16px system-ui,sans-serif;margin:2rem;background:#101a2d;color:#e9edf5}}
main{{max-width:1500px;margin:auto}}.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}}
figure{{margin:0;background:#fff;color:#17243c;padding:.5rem;border-radius:8px}}img{{width:100%}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#17243c;padding:1rem}}
</style><main><h1>{html.escape(title)}</h1>
<p>Experimental scientific evidence and explicitly artistic mappings. Apparent TEC-feature
motion is not plasma velocity. Unsupported channels are not quiet measurements.</p>
<div class="plots">{cards}</div><h2>Evidence and provenance</h2><pre>{source}</pre></main></html>'''
    (destination / "report.html").write_text(document, encoding="utf-8")


def science_diagnostics(dataset: Any, event: dict[str, Any], packet: dict[str, Any],
                        destination: Path) -> None:
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    from ophanim.dynamics.regularize import utc64
    frame_index = int(np.argmin(np.abs(dataset.time.values - utc64(event["time"]))))
    plots = {"tec": "tec", "background": "tec_background", "dtec": "dtec",
             "normalized_dtec": "dtec_z", "dtec_dt": "dtec_dt", "gradient": "grad_mag",
             "orientation": "structure_orientation", "coherence": "structure_coherence",
             "flow_speed": "flow_speed", "flow_confidence": "flow_confidence",
             "motion_eligibility": "flow_interpretation_confidence",
             "growth_ambiguity": "growth_ambiguity", "observation_support": "observation_support",
             "measurement_reliability": "measurement_reliability", "source_uncertainty": "source_uncertainty",
             "divergence": "flow_divergence", "vorticity": "flow_vorticity", "strain": "flow_strain",
             "advection_residual": "advection_residual", "residual_confidence": "advection_residual_confidence"}
    for filename, variable in plots.items():
        _field_plot(dataset, variable, destination / f"{filename}.png", frame_index=frame_index)
    figure, axes = _figure()
    if all(name in dataset for name in ("flow_u", "flow_v", "flow_confidence")):
        fields = [dataset[name].isel(time=frame_index) if "time" in dataset[name].dims else dataset[name]
                  for name in ("flow_u", "flow_v")]
        u, v = [np.asarray(field) for field in fields]
        confidence = np.zeros_like(u)
        if "flow_interpretation_confidence" in dataset:
            channel = dataset.flow_interpretation_confidence
            confidence = np.asarray(channel.isel(time=frame_index) if "time" in channel.dims else channel)
        stride = max(1, max(u.shape) // 20)
        from ophanim.dynamics.schemas import FlowConfig
        flow_settings = packet.get("config", {}).get("flow", {})
        minimum_confidence = float(flow_settings.get("minimum_confidence", FlowConfig().minimum_confidence))
        valid = np.isfinite(u) & np.isfinite(v) & (confidence > 0) & (confidence >= minimum_confidence)
        xx, yy = np.meshgrid(dataset.x.values / 1000, dataset.y.values / 1000)
        if valid[::stride, ::stride].any():
            axes.quiver(xx[::stride, ::stride], yy[::stride, ::stride],
                        np.where(valid, u, np.nan)[::stride, ::stride],
                        np.where(valid, v, np.nan)[::stride, ::stride])
        else:
            axes.text(.5, .5, "No eligible full-vector interpretation; raw estimates retained", transform=axes.transAxes, ha="center", fontsize=9)
        axes.text(.02, .02, f"Pair admission + supplied RMS + local score ≥ {minimum_confidence:g}; not validated accuracy",
                  transform=axes.transAxes, fontsize=8)
    axes.set(title="Apparent TEC-feature motion (not plasma velocity)", xlabel="x (km)", ylabel="y (km)")
    figure.savefig(destination / "flow.png", dpi=110)
    figure.clear()
    figure, axes = _figure()
    wave = event.get("wave") or {}
    spectrum = wave.get("spectrum_summary") or {}
    if len(spectrum.get("frequency_hz", [])) and len(spectrum.get("power", [])):
        order = np.argsort(spectrum["frequency_hz"])
        axes.plot(np.asarray(spectrum["frequency_hz"])[order], np.asarray(spectrum["power"])[order])
        axes.set(xlabel="Signed temporal frequency (Hz)", ylabel="Spectral power")
    else:
        axes.text(.5, .5, str(wave.get("status", "unavailable")), ha="center", va="center")
    axes.set_title("Wave spectrum / eligibility")
    figure.savefig(destination / "wave_spectrum.png", dpi=110)
    figure.clear()
    figure, axes = _figure()
    spatial = np.asarray(spectrum.get("spatial_power", []), dtype=float)
    kx, ky = np.asarray(spectrum.get("kx_rad_per_m", [])), np.asarray(spectrum.get("ky_rad_per_m", []))
    if spatial.ndim == 2 and spatial.shape == (len(ky), len(kx)) and len(kx) > 1 and len(ky) > 1:
        artist = axes.pcolormesh(kx*1000, ky*1000, np.log10(np.maximum(spatial, 1e-30)), shading="auto")
        figure.colorbar(artist, ax=axes, label="log₁₀ power · negative temporal frequencies")
        peak = spectrum.get("selected_peak") or {}
        if peak.get("kx_rad_per_m") is not None:
            axes.plot(peak["kx_rad_per_m"]*1000, peak["ky_rad_per_m"]*1000, "r+", markersize=12, label="Selected candidate (may be rejected)")
            axes.legend(fontsize=8)
        axes.set(xlabel="kx (rad/km)", ylabel="ky (rad/km)")
    else:
        axes.text(.5, .5, str(wave.get("status", "unavailable")), transform=axes.transAxes, ha="center")
    axes.set_title("Spatial spectrum and competing directions")
    figure.savefig(destination / "wave_spatial_spectrum.png", dpi=110)
    figure.clear()
    figure, axes = _figure()
    temporal = wave.get("temporal_support") or {}
    envelope = np.asarray(temporal.get("envelope_amplitude_tecu", []), dtype=float)
    if envelope.size:
        active = np.asarray(temporal.get("active_epoch_mask", []), dtype=bool)
        axes.plot(np.arange(len(envelope)), envelope, label="Spatially demodulated envelope")
        axes.axhline(temporal.get("amplitude_threshold_tecu", 0), linestyle="--", color="gray", label="Occupancy threshold")
        if active.shape == envelope.shape:
            axes.scatter(np.flatnonzero(active), envelope[active], s=10, label="Active epochs")
        axes.legend(fontsize=8)
        axes.set(xlabel="Analysis-window epoch", ylabel="TECU")
        axes.set_title(f"Temporal support · {temporal.get('effective_cycles', 0):.2f} effective cycles")
    else:
        axes.text(.5, .5, "Temporal carrier support unavailable", transform=axes.transAxes, ha="center")
        axes.set_title("Temporal support — not inferred from elapsed time alone")
    figure.savefig(destination / "wave_temporal_support.png", dpi=110)
    figure.clear()
    figure, axes = _figure()
    components = {key:value for key,value in (wave.get("confidence_components") or {}).items()
                  if isinstance(value, (int, float)) and np.isfinite(value)}
    if components:
        axes.barh([key.replace("_", " ") for key in components], list(components.values()))
    else:
        axes.text(.5, .5, "No admitted wave confidence components", transform=axes.transAxes, ha="center")
    axes.set(title="Wave evidence components — heuristic, not probabilities", xlim=(0, 1))
    figure.savefig(destination / "wave_confidence_components.png", dpi=110)
    figure.clear()
    figure, axes = _figure()
    scores = event.get("event_scores", event.get("scores", {}))
    if scores:
        axes.bar(list(scores), [float(value or 0) for value in scores.values()])
        axes.tick_params(axis="x", rotation=25)
    axes.set(title="Heuristic event scores — not probabilities", ylim=(0, 1))
    figure.savefig(destination / "event_summary.png", dpi=110)
    figure.clear()
    extra_images = ["flow.png", "wave_spectrum.png", "wave_spatial_spectrum.png", "wave_temporal_support.png",
                    "wave_confidence_components.png", "event_summary.png"]
    timeline_path = destination.parent / "events.json"
    if packet.get("event_timeline") and timeline_path.is_file():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        frames = timeline.get("frames", [])
        figure, axes = _figure()
        if frames:
            for name in ("quiet", "flow", "wave", "front", "localized", "disturbed", "uncertain"):
                axes.plot(range(len(frames)), [item["event"].get("event_scores", {}).get(name, np.nan) for item in frames], label=name)
            axes.legend(fontsize=8, ncol=4)
            axes.set_xticks(range(len(frames)), [item["time"][11:19] for item in frames], rotation=40)
        axes.set(title="Exact-time event interpretations (gaps are not quiet)", ylabel="Independent heuristic score", xlabel="Selected UTC epochs", ylim=(0, 1))
        figure.savefig(destination / "event_timeline.png", dpi=110)
        figure.clear()
        extra_images.append("event_timeline.png")
    _report(destination, "OPHANIM scientific diagnostics",
            [f"{name}.png" for name in plots] + extra_images, packet)


def visual_diagnostics(dataset: Any, destination: Path, *, frame_index: int = -1) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = {"base_density": "base_density", "emission": "emission", "hue_driver": "hue_driver",
             "opacity": "base_opacity", "lic": "flow_texture", "wave_modulation": "wave_modulation",
             "height_displacement": "height_displacement", "breakup": "breakup"}
    names.update(front_fold="front_fold", ribbon_strength="ribbon_strength",
                 fiber_strength="flow_texture_strength", display_emission="display_emission")
    for filename, variable in names.items():
        _field_plot(dataset, variable, destination / f"{filename}.png", frame_index=frame_index)
    _report(destination, "ShawtyNet artistic mapping diagnostics",
            [f"{name}.png" for name in names], dataset.attrs)
