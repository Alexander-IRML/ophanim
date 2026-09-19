"""Deterministic native-grid discovery fixture; never an observation product."""


def make_discovery_demo(seed=42):
    import numpy as np
    import xarray as xr

    lat = np.arange(-87.5, 90, 2.5)
    lon = np.arange(-180, 180, 5.0)
    time = np.datetime64("2024-05-01T00", "ns") + np.arange(193) * np.timedelta64(1, "h")
    hour = np.arange(len(time))[:, None, None]
    daylight = np.cos(2 * np.pi * (hour / 24 + lon[None, None, :] / 360 - .5))
    tec = 18 + 8 * daylight * np.cos(np.radians(lat))[None, :, None]
    tec = tec + np.random.default_rng(seed).normal(0, .15, tec.shape)
    # Persistent independent events during the final day, on original cells.
    tex = (lat[:, None] >= 27.5) & (lat[:, None] <= 35) & (lon[None, :] >= -105) & (lon[None, :] <= -95)
    pac = (lat[:, None] >= -30) & (lat[:, None] <= -22.5) & (lon[None, :] >= 145) & (lon[None, :] <= 155)
    tec[180:190, tex] += 16
    tec[182:192, pac] -= 10
    return xr.Dataset(
        {"tec": (("time", "lat", "lon"), tec, {"units": "TECU", "semantic_class": "synthetic"}),
         "observed_mask": (("time", "lat", "lon"), np.ones_like(tec, dtype=bool)),
         "source_rms_tecu": (("time", "lat", "lon"), np.full_like(tec, .15)),
         "available_at": ("time", time)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source_kind": "synthetic", "source_metadata": {
            "source_kind": "synthetic", "latitude_step_degrees": 2.5,
            "longitude_step_degrees": 5., "cadence_seconds": 3600,
            "source_ids": [f"discovery-demo-v1:{seed}"],
        }, "synthetic_truth": {"seed": seed, "positive_tecu": 16, "negative_tecu": -10},
        "semantics": "Planted mathematical events for interface exploration; not actual space weather"},
    )
