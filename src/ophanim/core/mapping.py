"""JSON-safe native map views; no interpolation or artistic styling."""

from __future__ import annotations


def native_map(dataset, *, values=None, field="tec", epoch_index=-1):
    """Return a small transport contract retaining native cell coordinates."""
    import numpy as np

    if not dataset.sizes.get("time"):
        return {"latitudes": [], "longitudes": [], "values": [], "field": field, "epoch": None}
    frame = np.asarray(dataset.tec.isel(time=epoch_index).values if values is None else values, dtype=float)
    return {
        "latitudes": [float(v) for v in dataset.lat.values],
        "longitudes": [float(v) for v in dataset.lon.values],
        "values": [[float(v) if np.isfinite(v) else None for v in row] for row in frame],
        "field": field,
        "units": "TECU",
        "epoch": np.datetime_as_string(dataset.time.values[epoch_index], unit="s") + "Z",
        "sampling": "native source nodes; not independent instrument measurements",
    }
