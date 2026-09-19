"""Manual photograph compositing with foreground/cloud masks and grading."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path

from ophanim.shawtynet.mapping import canonical_json


def _srgb_to_linear(values):
    import numpy as np
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values):
    import numpy as np
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055)


def composite_arrays(
    photograph, render_rgba, *, foreground_mask=None, cloud_mask=None,
    horizon_y: float = 0.5, horizon_fade: float = 0.6,
    saturation: float = 0.85, exposure: float = 1.0,
    grade_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Blend normalized sRGB arrays in linear light; return RGB in [0,1].

    Render alpha is straight (unassociated). Masks range from zero (clear) to
    one (fully occluding). ``horizon_y`` is a fraction from the image top;
    rendered structure is excluded below that manually chosen horizon.
    These atmospheric controls are artistic, not an atmospheric simulation.
    """
    import numpy as np

    photo, render = np.asarray(photograph, dtype=float), np.asarray(render_rgba, dtype=float)
    if photo.ndim != 3 or photo.shape[-1] != 3 or render.shape != (*photo.shape[:2], 4):
        raise ValueError("photograph RGB and render RGBA dimensions must match")
    if min(photo.shape[:2]) < 1:
        raise ValueError("photograph cannot be empty")
    if any(not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)) for array in (photo, render)):
        raise ValueError("image values must be finite normalized sRGB")
    for name, value, low, high in (
        ("horizon_y", horizon_y, 0.001, 1), ("horizon_fade", horizon_fade, 0, 1),
        ("saturation", saturation, 0, 2), ("exposure", exposure, 0, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
    if not isinstance(grade_rgb, (tuple, list)) or len(grade_rgb) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10 for value in grade_rgb):
        raise ValueError("grade_rgb needs three finite gains in [0,10]")
    alpha = render[..., 3].copy()
    for mask in (foreground_mask, cloud_mask):
        if mask is None:
            continue
        values = np.asarray(mask, dtype=float)
        if values.shape != photo.shape[:2] or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("occlusion masks must match image dimensions and range [0,1]")
        alpha *= 1 - values
    rows = (np.arange(photo.shape[0]) + 0.5) / photo.shape[0]
    distance_above = (horizon_y - rows) / horizon_y
    # A continuous C1 taper reaches zero AT the manually chosen horizon.
    # Previously the default jumped from 40% transmission to zero there.
    # Even "no atmospheric fading" retains a two-pixel antialiased cutoff.
    fade_width = max(2 / photo.shape[0] / horizon_y, 0.04 + 0.46 * horizon_fade)
    fraction = np.clip(distance_above / fade_width, 0, 1)
    atmospheric_gain = fraction * fraction * (3 - 2 * fraction)
    alpha *= atmospheric_gain[:, None]
    foreground = _srgb_to_linear(render[..., :3])
    luminance = np.sum(foreground * np.array([0.2126, 0.7152, 0.0722]), axis=-1, keepdims=True)
    # Saturation gradually drops toward the horizon independently of alpha.
    local_saturation = saturation * (0.65 + 0.35 * np.clip(distance_above, 0, 1))
    foreground = luminance + local_saturation[:, None, None] * (foreground - luminance)
    foreground = np.maximum(foreground, 0) * exposure * np.asarray(grade_rgb)
    background = _srgb_to_linear(photo)
    result = foreground * alpha[..., None] + background * (1 - alpha[..., None])
    return np.clip(_linear_to_srgb(result), 0, 1).astype(np.float32)


def composite_photograph(
    photo_path: str | Path, render_path: str | Path, output_path: str | Path, *,
    foreground_mask: str | Path | None = None, cloud_mask: str | Path | None = None,
    horizon_y: float = 0.5, horizon_fade: float = 0.6,
    saturation: float = 0.85, exposure: float = 1.0,
    grade_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Composite one manually aligned photo/render and save an audit receipt.

    Photograph and render must already have matching pixel dimensions. White
    in a foreground/cloud mask hides the shell; gray clouds partly transmit.
    Automatic camera calibration and foreground segmentation are out of scope.
    """
    import numpy as np
    from PIL import Image, ImageOps

    photo_file, render_file, output = Path(photo_path), Path(render_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"composite already exists: {output}")
    if output.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("composite output must be PNG or JPEG")
    with Image.open(photo_file) as opened:
        photo = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=float) / 255
    with Image.open(render_file) as opened:
        if "A" not in opened.getbands():
            raise ValueError("render must include an alpha channel")
        render = np.asarray(opened.convert("RGBA"), dtype=float) / 255
    masks = []
    for file in (foreground_mask, cloud_mask):
        if file is None:
            masks.append(None)
        else:
            with Image.open(file) as opened:
                masks.append(np.asarray(opened.convert("L"), dtype=float) / 255)
    result = composite_arrays(
        photo, render, foreground_mask=masks[0], cloud_mask=masks[1],
        horizon_y=horizon_y, horizon_fade=horizon_fade, saturation=saturation,
        exposure=exposure, grade_rgb=grade_rgb,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.rint(result * 255).astype(np.uint8))
    image.save(output, **({"quality": 95, "subsampling": 0} if output.suffix.lower() in (".jpg", ".jpeg") else {}))
    inputs = {"photograph": photo_file, "render": render_file, "foreground_mask": foreground_mask, "cloud_mask": cloud_mask}
    receipt = {
        "schema_version": "ophanim-composite/1", "category": "ARTISTIC",
        "inputs": {name: {"filename": Path(file).name, "sha256": sha256(Path(file).read_bytes()).hexdigest()} for name, file in inputs.items() if file is not None},
        "settings": {"horizon_y": horizon_y, "horizon_fade": horizon_fade, "saturation": saturation, "exposure": exposure, "grade_rgb": grade_rgb},
        "output_sha256": sha256(output.read_bytes()).hexdigest(),
        "semantic": "Manually aligned artistic composite; atmospheric fading and color grading are artistic controls.",
    }
    output.with_suffix(".composite.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return output
