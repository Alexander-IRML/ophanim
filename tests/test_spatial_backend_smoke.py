"""Opt-in end-to-end smoke test for the CPU spatial backend.

The release gate runs this explicitly inside the Docker image.  It is opt-in
because even the deliberately small twenty-day fixture performs real ConvLSTM
optimization on the native 71 by 72 grid.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from ophanim.mamba_monitor import MonitorRegion, NativeGridDay, NativeGridSnapshot
from ophanim.spatial_monitor import TorchSpatialBackend


RUN_SMOKE = os.environ.get("OPHANIM_RUN_SPATIAL_TRAIN_SMOKE") == "1"


@unittest.skipUnless(RUN_SMOKE, "set OPHANIM_RUN_SPATIAL_TRAIN_SMOKE=1")
class TorchSpatialBackendSmokeTests(unittest.TestCase):
    def test_real_training_artifact_and_incremental_scan(self) -> None:
        import numpy as np

        start = date(2026, 1, 1)
        days = tuple(self._day(index, start) for index in range(21))
        archive = _SyntheticArchive(days, np=np)
        training_snapshot = NativeGridSnapshot(
            pipeline_id="mamba-pipeline-smoke",
            model_version_id="mamba-model-smoke",
            sync_through_date=days[19].product_date,
            source_cursor=days[19].source_day_id,
            days=days[:20],
        )
        progress_events: list[tuple[str, int, int, str]] = []
        backend = TorchSpatialBackend(
            batch_size=16,
            checkpoint_interval=50,
            normalization_sample_cells=20_000,
            calibration_sample_cells=20_000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "spatial-checkpoint.json"
            result = backend.train(
                snapshot=training_snapshot,
                archive=archive,  # type: ignore[arg-type]
                region=MonitorRegion(),
                options={"epochs": 1},
                checkpoint_path=checkpoint,
                progress=lambda *event: progress_events.append(event),
                interrupted=lambda: False,
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(result.completed_epochs, 1)
            self.assertEqual(result.training_window_count, 156)
            self.assertEqual(result.validation_window_count, 12)
            self.assertEqual(result.calibration_window_count, 12)
            self.assertEqual(result.test_window_count, 12)
            self.assertGreater(len(result.model_content), 100)
            self.assertTrue(progress_events)

            scan_snapshot = replace(
                training_snapshot,
                sync_through_date=days[20].product_date,
                source_cursor=days[20].source_day_id,
                days=days,
            )
            scan = backend.scan(
                model_content=result.model_content,
                model_metadata={
                    "training_data_hash": result.training_data_hash,
                    "training_end_at": result.training_end_at.isoformat(),
                    "threshold": result.threshold,
                    "residual_center_tecu": result.residual_center_tecu,
                    "residual_scale_tecu": result.residual_scale_tecu,
                },
                snapshot=scan_snapshot,
                archive=archive,  # type: ignore[arg-type]
                after_source_cursor=days[19].source_day_id,
                region=MonitorRegion(),
                progress=lambda *_event: None,
                interrupted=lambda: False,
            )
            self.assertEqual(scan.map_count, 12)
            self.assertEqual(scan.usable_map_count, 12)
            self.assertIsNotNone(scan.max_anomaly_score)

    @staticmethod
    def _day(index: int, start: date) -> NativeGridDay:
        return NativeGridDay(
            source_day_id=index + 1,
            product_date=start + timedelta(days=index),
            grid_set_id=sha256(f"grid-{index}".encode()).hexdigest(),
            checksum_sha256=sha256(f"source-{index}".encode()).hexdigest(),
            revision_priority=20,
        )


class _SyntheticArchive:
    def __init__(self, days: tuple[NativeGridDay, ...], *, np: object) -> None:
        self._days = {day.grid_set_id: (index, day) for index, day in enumerate(days)}
        self._np = np
        self._latitudes = np.asarray(
            [index * 2.5 - 87.5 for index in range(71)], dtype=np.float64
        )
        self._longitudes = np.asarray(
            [index * 5.0 - 180.0 for index in range(72)], dtype=np.float64
        )

    def read_native(self, grid_set_id: str) -> object:
        np = self._np
        day_index, day = self._days[grid_set_id]
        origin = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        times = [origin + timedelta(hours=2 * index) for index in range(13)]
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        times_us = np.asarray(
            [int((value - epoch).total_seconds() * 1_000_000) for value in times],
            dtype=np.int64,
        )
        latitude_signal = self._latitudes[:, None] * 0.002
        longitude_signal = self._longitudes[None, :] * 0.001
        values = np.stack(
            [
                np.asarray(
                    12.0
                    + day_index * 0.03
                    + time_index * 0.015
                    + latitude_signal
                    + longitude_signal,
                    dtype=np.float32,
                )
                for time_index in range(13)
            ],
            axis=0,
        )
        valid = np.ones(values.shape, dtype=np.bool_)
        return SimpleNamespace(
            manifest=SimpleNamespace(
                grid_set_id=day.grid_set_id,
                source_checksum_sha256=day.checksum_sha256,
                revision_priority=day.revision_priority,
                shape=values.shape,
            ),
            times_utc_microseconds=times_us,
            latitudes_degrees=self._latitudes,
            longitudes_degrees=self._longitudes,
            vtec_tecu=values,
            valid=valid,
        )


if __name__ == "__main__":
    unittest.main()
