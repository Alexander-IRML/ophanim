"""Deterministic CPU line-integral convolution for artistic flow fibers."""

from __future__ import annotations


def line_integral_convolution(
    u, v, valid, *, dx_m: float, dy_m: float, seed: int,
    steps: int = 25, step_cells: float = 0.75, contrast: float = 1.8,
    minimum_speed: float = 0.01, seed_noise=None,
):
    """Return [0,1] seeded texture; invalid/low-speed cells are zero.

    Velocity components are metres/second, columns point east, and rows point
    north. Streamlines terminate at invalid support or the image boundary.
    Texture is an artistic transformation, never an inferred observation.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if u.ndim != 2 or v.shape != u.shape or valid.shape != u.shape:
        raise ValueError("LIC requires same-shaped two-dimensional flow and mask")
    if min(u.shape) < 2 or dx_m <= 0 or dy_m <= 0:
        raise ValueError("LIC requires a nondegenerate metric grid")
    if not isinstance(steps, int) or steps < 1 or not 0 < step_cells <= 1 or contrast <= 0:
        raise ValueError("invalid LIC streamline settings")
    usable = valid & np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) > minimum_speed)
    px = np.where(usable, u / dx_m, 0.0)
    py = np.where(usable, v / dy_m, 0.0)
    norm = np.hypot(px, py)
    px = np.divide(px, norm, out=np.zeros_like(px), where=norm > 0)
    py = np.divide(py, norm, out=np.zeros_like(py), where=norm > 0)
    noise = np.random.default_rng(seed).random(u.shape) if seed_noise is None else np.asarray(seed_noise, dtype=float)
    if noise.shape != u.shape or not np.all(np.isfinite(noise)) or np.any((noise < 0) | (noise > 1)):
        raise ValueError("LIC seed noise must be finite, same-shaped and in [0,1]")
    accumulated = noise.copy()
    weight_sum = np.ones(u.shape)
    origin_y, origin_x = np.indices(u.shape, dtype=float)
    for direction in (-1, 1):
        xpos, ypos = origin_x.copy(), origin_y.copy()
        active = usable.copy()
        for index in range(1, steps + 1):
            coords = np.array([ypos, xpos])
            local_x = map_coordinates(px, coords, order=1, mode="nearest")
            local_y = map_coordinates(py, coords, order=1, mode="nearest")
            local_norm = np.hypot(local_x, local_y)
            active &= local_norm > 1e-8
            local_x = np.divide(local_x, local_norm, out=np.zeros_like(local_x), where=local_norm > 0)
            local_y = np.divide(local_y, local_norm, out=np.zeros_like(local_y), where=local_norm > 0)
            xpos += direction * step_cells * local_x
            ypos += direction * step_cells * local_y
            active &= (xpos >= 0) & (xpos <= u.shape[1] - 1) & (ypos >= 0) & (ypos <= u.shape[0] - 1)
            coords = np.array([ypos, xpos])
            # Require every contributing support pixel; a streamline cannot
            # bridge a missing-data hole via bilinear interpolation.
            active &= map_coordinates(usable.astype(float), coords, order=1, mode="constant", cval=0) >= 1 - 1e-8
            weight = np.exp(-0.5 * (index / max(1, steps / 2)) ** 2)
            sampled = map_coordinates(noise, coords, order=1, mode="nearest")
            accumulated += np.where(active, weight * sampled, 0)
            weight_sum += np.where(active, weight, 0)
    texture = np.clip((accumulated / weight_sum - 0.5) * contrast + 0.5, 0, 1)
    return np.where(usable, texture, 0).astype(np.float32)
