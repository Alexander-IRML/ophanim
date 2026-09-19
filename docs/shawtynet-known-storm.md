# Real known-event check: May 10–11, 2024

This is a **qualitative, retrospective sanity check**, not a measurement of
real-world detection accuracy. Unlike the artistic photo demonstration, its TEC
values are actual archived source data, not synthetic fields with historical dates.

## Independent event identification

NOAA reported that G5 conditions first occurred at 18:54 EDT on May 10, 2024
(22:54 UTC). [NOAA Space Weather Prediction Center](https://www.swpc.noaa.gov/news/g5-conditions-observed).
USGS independently described storm commencement at 12:06 Eastern time that day
(16:06 UTC). These timestamps describe different aspects of the event, not two
estimates of the same onset. [USGS event account](https://www.usgs.gov/programs/geomagnetism/science/may-10-2024-magnetic-disturbance).

## Reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-shawty/bin/python scripts/shawtynet_known_storm.py
```

Use `--fetch-only` to acquire the inputs without analyzing, or `--output PATH`
to choose a separate validation directory. The script uses the existing trusted
CODE downloader. It downloads three final daily products from May 9–11, about
1.08 MB compressed in total, and keeps source checksums, URLs and actual local
acquisition times. It does **not** write into the desktop SQLite database, the
native OPHANIM Zarr archive, or application configuration. Subsequent runs verify
and reuse the downloaded bytes and matching completed scientific run.

The local acquisition timestamps are in 2026 and are not invented historical
publication times. Consequently, this final-product analysis is retrospective
and must never be represented as an operational May 2024 forecast.

The source products are:

- [CODE final May 9, 2024](https://www.aiub.unibe.ch/download/CODE/2024/COD0OPSFIN_20241300000_01D_01H_GIM.INX.gz)
- [CODE final May 10, 2024](https://www.aiub.unibe.ch/download/CODE/2024/COD0OPSFIN_20241310000_01D_01H_GIM.INX.gz)
- [CODE final May 11, 2024](https://www.aiub.unibe.ch/download/CODE/2024/COD0OPSFIN_20241320000_01D_01H_GIM.INX.gz)

The broad Texas-region selection is 20–42°N, 112–88°W. It contains 9 latitude
nodes by 5 longitude nodes on the original **2.5° latitude × 5° longitude** grid.
The sequence contains 73 hourly epochs, May 9 00:00 through May 12 00:00 UTC.
At duplicate midnight epochs, the later daily final product wins; the selected
source ID for every timestamp is recorded.

Science uses a 75 km local projected grid, 100 km smoothing, a 200 km structure
tensor scale, and a centered 12-hour rolling-median background. The event target
is May 10 at 23:00 UTC, inside full baseline support. The 73 × 30 × 29 analysis
cube contains 63,510 cells; the script imposes a 200,000-cell budget. Neither
interpolation nor smoothing creates finer source resolving power.

## Recorded result and visual review

The run completed and its native/regional time series and target dTEC map were
visually inspected on September 18, 2026. Outputs on this laptop:

- [Time-series comparison](../var/shawtynet-validation/known-storm/known-storm-timeseries.png)
- [Machine-readable summary and source provenance](../var/shawtynet-validation/known-storm/summary.json)
- [Full scientific diagnostics](../var/shawtynet-validation/known-storm/science/a360df692932737b3263dbaa/diagnostics/report.html)

| Descriptive quantity | May 9 comparison | May 10 event interval |
|---|---:|---:|
| Native area-weighted regional mean TEC at 22:00 UTC | 64.40 TECU | 122.56 TECU |
| Regional mean TEC averaged over 18:00–23:00 UTC | 63.08 TECU | 99.55 TECU |
| Spatial P95 absolute 12-hour median residual, averaged over those six epochs | 10.62 TECU | 37.75 TECU |

The largest positive same-UT difference during the independently identified
storm interval is **+58.16 TECU at May 10 22:00 UTC**. The reviewed time series
shows a pronounced regional enhancement after the USGS commencement timestamp,
near the first NOAA G5 report, followed by a lower regional level on May 11.
The target map shows a broad spatial gradient in a positive median-relative
residual rather than fabricated fine-scale wave structure. These are qualitatively
sensible changes for a known-disturbed interval; the temporal correspondence is
an observation, **not a physical attribution or proof of detector accuracy**.

At the selected target the baseline is available. The event heuristic returns
`front_like`, with score-derived confidence 0.359; this label only describes the
map morphology and is not an identified physical traveling front. Actual flow is
withheld as `invalid_support` because the finite-pair support gate fails after
preprocessing. The wave channel correctly returns `insufficient_samples`: hourly
native data cannot support the configured 30–180 minute period search with six
actual samples per period. No gate was weakened to produce an attractive result.

## What this establishes—and what it does not

The complete native-input → scientific-state → diagnostics path runs on an
independently identified real disturbed interval. Large-scale TEC changes survive
preprocessing, intermediate values are inspectable, and unsupported flow/wave
claims are withheld. This addresses the brief's **qualitative** known-disturbance
check at the scale supported by this source.

May 9 is one preceding-day, same-UTC comparison, **not an independently verified
quiet control** or a climatology. The centered median is not a quiet-day storm
baseline and can retain normal diurnal structure. We have not established a
false-positive rate, a detection probability, a causal mechanism, a calibrated
event confidence, a physical velocity, or the validity of fine-scale wave
inference on real observations. Independent quiet/event collections and
better-resolved TEC sources remain necessary for those claims.
