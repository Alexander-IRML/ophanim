# Changelog

All notable changes to OPHANIM are recorded here. The project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-16

First experimental deployment:

- dependency-free local TEC ingestion, regional aggregation, forecasting, and
  disturbance-candidate workflow;
- automatic CODE GIM acquisition and fine regional readout tables;
- PostgreSQL catalog plus immutable native and derived Zarr storage;
- resumable twenty-year selective state-space (Mamba-style) baseline;
- causal native-grid ConvLSTM baseline with side-by-side anomaly comparison;
- loopback-only desktop application that prefers Windows Google Chrome, plus a
  Docker Compose deployment.

This release is a research experiment, not an operational space-weather alert
service.
