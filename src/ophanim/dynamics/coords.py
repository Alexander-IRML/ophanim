"""One explicit geographic-to-metric boundary for all numerical derivatives."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
from pyproj import CRS, Geod, Proj, Transformer
from scipy.interpolate import RegularGridInterpolator

from .schemas import AnalysisConfig, AnalysisError


def source_metadata(data: xr.Dataset) -> dict:
    value = data.attrs.get("source_metadata", {})
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise AnalysisError("source_metadata must be an object")
    result = dict(value)
    result.setdefault("source_kind", data.attrs.get("source_kind", "unknown"))
    return result


def _spacing(values: np.ndarray, name: str) -> float:
    delta = np.diff(values)
    if len(delta) < 1 or not np.isfinite(values).all() or (delta <= 0).any():
        raise AnalysisError(f"{name} must be finite, strictly increasing with at least two points")
    spacing = float(np.median(delta))
    if not np.allclose(delta, spacing, rtol=1e-5, atol=1e-5):
        raise AnalysisError(f"{name} must have regular metric spacing")
    return spacing


def project_dataset(data: xr.Dataset, config: AnalysisConfig) -> xr.Dataset:
    """Regrid to a regular local AEQD surface in metres, without extrapolation.

    Geographic inputs must be regional. x/y are projected axes (approximately
    east/north near the projection center), not global constant-bearing axes.
    The distortion limit prevents quietly applying this local model globally.
    """
    meta = source_metadata(data)
    if data.tec.dims == ("time", "y", "x"):
        result = data.sortby("x").sortby("y").copy(deep=True)
        dx = _spacing(result.x.values, "x")
        dy = _spacing(result.y.values, "y")
        if result.sizes["x"] * result.sizes["y"] > config.max_grid_cells:
            raise AnalysisError("projected grid exceeds max_grid_cells")
        if result.tec.size > config.max_cube_cells:
            raise AnalysisError("projected sequence exceeds max_cube_cells")
        if meta.get("source_kind") in ("native", "synthetic"):
            meta.setdefault("native_dx_m", dx)
            meta.setdefault("native_dy_m", dy)
        center_lat = config.projection_center_lat if config.projection_center_lat is not None else float(meta.get("projection_center_lat", 30.0))
        center_lon = config.projection_center_lon if config.projection_center_lon is not None else float(meta.get("projection_center_lon", -98.0))
        if "lat" not in result.coords or "lon" not in result.coords:
            crs = CRS.from_proj4(f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs")
            inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xx, yy = np.meshgrid(result.x.values, result.y.values)
            lon, lat = inverse.transform(xx, yy)
            result = result.assign_coords(lat=(("y", "x"), lat), lon=(("y", "x"), lon))
        if "spatial_interpolated" in result:
            if result.spatial_interpolated.dims != result.tec.dims or result.spatial_interpolated.dtype.kind != "b":
                raise AnalysisError("spatial_interpolated must be a boolean mask matching the metric TEC grid")
            # A metric coordinate system does not turn an earlier interpolation
            # into a new independent observation. Sorting above also sorted this
            # mask, so preserve the caller's field-level provenance verbatim.
            result["spatial_interpolated"] = result.spatial_interpolated.copy(deep=True)
        else:
            result["spatial_interpolated"] = xr.zeros_like(result.observed_mask, dtype=bool)
        result.attrs["source_metadata"] = meta
        result.attrs.setdefault("projection", "provided metric grid; axes x eastward/y northward")
    else:
        if data.lat.ndim != 1 or data.lon.ndim != 1:
            raise AnalysisError("geographic input requires one-dimensional latitude/longitude axes")
        lat = np.asarray(data.lat.values, float)
        lon = np.asarray(data.lon.values, float)
        if min(len(lat), len(lon)) < 2 or not np.isfinite(lat).all() or not np.isfinite(lon).all() or (np.abs(lat) > 90).any():
            raise AnalysisError("projection requires at least two valid latitudes and longitudes")
        center_lat = config.projection_center_lat if config.projection_center_lat is not None else float((lat.min() + lat.max()) / 2)
        center_lon = config.projection_center_lon if config.projection_center_lon is not None else float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(lon))))))
        unwrapped = center_lon + (lon - center_lon + 180) % 360 - 180
        data = data.assign_coords(lon=unwrapped).sortby("lat").sortby("lon")
        lat, lon = data.lat.values, data.lon.values
        if (np.diff(lat) <= 0).any() or (np.diff(lon) <= 0).any():
            raise AnalysisError("geographic axes must not have duplicate coordinates")
        xx_lon, yy_lat = np.meshgrid(lon, lat)
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs")
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        native_x, native_y = forward.transform(xx_lon, yy_lat)
        radius = np.hypot(native_x, native_y)
        if not np.isfinite(radius).all() or np.max(radius) > config.max_projection_radius_km * 1000:
            raise AnalysisError("region exceeds local projection radius; select a regional analysis window")
        factors = Proj(crs).get_factors(xx_lon, yy_lat)
        scale_error = float(np.max(np.maximum(np.abs(np.asarray(factors.tissot_semimajor) - 1), np.abs(np.asarray(factors.tissot_semiminor) - 1))))
        if scale_error > config.max_projection_scale_error:
            raise AnalysisError("projection distortion exceeds configured scale-error limit")
        if meta.get("source_kind") == "native":
            meta.setdefault("native_lat_spacing_deg", float(np.median(np.diff(lat))))
            meta.setdefault("native_lon_spacing_deg", float(np.median(np.diff(lon))))
            geod = Geod(ellps="WGS84")
            meta.setdefault("native_dx_m", abs(float(geod.inv(center_lon, center_lat, center_lon + meta["native_lon_spacing_deg"], center_lat)[2])))
            meta.setdefault("native_dy_m", abs(float(geod.inv(center_lon, center_lat, center_lon, min(89.999, center_lat + meta["native_lat_spacing_deg"]))[2])))
        spacing = config.science_spacing_km * 1000
        x = np.arange(np.ceil(native_x.min() / spacing), np.floor(native_x.max() / spacing) + 1) * spacing
        y = np.arange(np.ceil(native_y.min() / spacing), np.floor(native_y.max() / spacing) + 1) * spacing
        if min(len(x), len(y)) < 3:
            raise AnalysisError("region is smaller than three analysis-grid points; reduce numerical spacing or enlarge region")
        if len(x) * len(y) > config.max_grid_cells:
            raise AnalysisError("analysis grid exceeds max_grid_cells; enlarge numerical spacing")
        if len(x) * len(y) * data.sizes["time"] > config.max_cube_cells:
            raise AnalysisError("projected sequence exceeds max_cube_cells; reduce window or increase spacing")
        xx, yy = np.meshgrid(x, y)
        target_lon, target_lat = inverse.transform(xx, yy)
        query_lon = center_lon + (target_lon - center_lon + 180) % 360 - 180
        points = np.stack((target_lat, query_lon), axis=-1)
        result = xr.Dataset(coords={"time": data.time.values, "x": x, "y": y, "lat": (("y", "x"), target_lat), "lon": (("y", "x"), target_lon)}, attrs=dict(data.attrs))
        masks = []
        for frame in data.observed_mask.values:
            weights = RegularGridInterpolator((lat, lon), frame.astype(float), bounds_error=False, fill_value=0)(points)
            masks.append(weights >= 1 - 1e-8)
        observed = np.asarray(masks)
        for name, variable in data.data_vars.items():
            if variable.dims == ("time", "lat", "lon") and name != "observed_mask":
                # Quality flags are bitsets and are not suitable for interpolation.
                if name == "quality_mask":
                    continue
                frames = []
                for index, frame in enumerate(variable.values):
                    values = RegularGridInterpolator((lat, lon), frame, bounds_error=False, fill_value=np.nan)(points)
                    frames.append(np.where(observed[index], values, np.nan))
                result[name] = (("time", "y", "x"), np.asarray(frames), dict(variable.attrs))
            elif variable.dims == ("time",):
                result[name] = variable.copy()
        result["observed_mask"] = (("time", "y", "x"), observed)
        result["spatial_interpolated"] = (("time", "y", "x"), observed.copy())
        result.attrs.update(projection=crs.to_string(), projection_center_lat=center_lat, projection_center_lon=center_lon, projection_max_scale_error=scale_error, source_metadata=meta)
    result.x.attrs.update(units="m", semantic="local eastward projected distance")
    result.y.attrs.update(units="m", semantic="local northward projected distance")
    result.lat.attrs["units"] = "degrees_north"
    result.lon.attrs["units"] = "degrees_east"
    result = result.assign_coords(x_km=("x", result.x.values / 1000), y_km=("y", result.y.values / 1000))
    return result


def flow_pixels_to_enu(row_displacement, column_displacement, dx_m: float, dy_m: float, dt_s: float):
    """Convert library (row,column) flow on ascending y/x axes to m/s.

    Arrays are intentionally ordered south-to-north: increasing array row is
    northward, unlike a conventional north-up image display. No hidden y flip.
    """
    if min(dx_m, dy_m, dt_s) <= 0:
        raise AnalysisError("flow spacing and interval must be positive")
    return np.asarray(column_displacement) * dx_m / dt_s, np.asarray(row_displacement) * dy_m / dt_s
