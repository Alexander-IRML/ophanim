"""End-to-end tests for the runnable OPHANIM v0 composition."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Barrier, Event, Lock, local

from ophanim.aggregation import (
    AggregationPolicy,
    BoundsRegionSelector,
    GeographicBounds,
    MeanMedianRegionalAggregator,
    boundary_checksum_sha256,
)
from ophanim.bootstrap import create_sqlite_application
from ophanim.domain import (
    DetectorVersion,
    DisturbanceAssessment,
    ForecastMode,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    TargetMetric,
)
from ophanim.forecasting import (
    ForecastRequest,
    ForecastingError,
    InsufficientHistoryError,
    PersistenceModelRunner,
)
from ophanim.ingestion import IngestionRequest
from ophanim.ionex import IONEXV1Parser
from ophanim.sqlite import SQLiteUnitOfWork
from ophanim.workflows import (
    ForecastConflictError,
    ForecastIssuingWorkflow,
    RegionalAggregationWorkflow,
    VerticalSliceRequest,
)


UTC = timezone.utc
ORIGIN = datetime(2024, 1, 2, 0, tzinfo=UTC)
VALID_AT = ORIGIN + timedelta(hours=2)
RUN_AT = datetime(2024, 1, 3, 12, tzinfo=UTC)


def _record(payload: str, label: str) -> str:
    return f"{payload:<60}{label:<20}\n"


def _grid_payload(*values: float) -> str:
    return "  " + "".join(f"{value:6.1f}" for value in values)


def _data_record(*values: int) -> str:
    return "".join(f"{value:5d}" for value in values).ljust(80) + "\n"


def _map(number: int, hour: int, value: int) -> str:
    lines = [
        _record(f"{number:6d}", "START OF TEC MAP"),
        _record(
            f"{2024:6d}{1:6d}{2:6d}{hour:6d}{0:6d}{0:6d}",
            "EPOCH OF CURRENT MAP",
        ),
    ]
    for latitude in (32.5, 31.5):
        lines.append(
            _record(
                _grid_payload(latitude, -100.0, -98.0, 1.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
        )
        lines.append(_data_record(value, value, value))
    lines.append(_record(f"{number:6d}", "END OF TEC MAP"))
    return "".join(lines)


def _two_epoch_ionex() -> str:
    header = "".join(
        (
            _record(f"{1.0:8.1f}{'':12}I{'':39}", "IONEX VERSION / TYPE"),
            _record(f"{2:6d}", "# OF MAPS IN FILE"),
            _record(f"{2:6d}", "MAP DIMENSION"),
            _record(_grid_payload(450.0, 450.0, 0.0), "HGT1 / HGT2 / DHGT"),
            _record(_grid_payload(32.5, 31.5, -1.0), "LAT1 / LAT2 / DLAT"),
            _record(_grid_payload(-100.0, -98.0, 1.0), "LON1 / LON2 / DLON"),
            _record(f"{-1:6d}", "EXPONENT"),
            _record("", "END OF HEADER"),
        )
    )
    return header + _map(1, 0, 100) + _map(2, 2, 200) + _record("", "END OF FILE")


def _one_epoch_ionex() -> str:
    header = "".join(
        (
            _record(f"{1.0:8.1f}{'':12}I{'':39}", "IONEX VERSION / TYPE"),
            _record(f"{1:6d}", "# OF MAPS IN FILE"),
            _record(f"{2:6d}", "MAP DIMENSION"),
            _record(_grid_payload(450.0, 450.0, 0.0), "HGT1 / HGT2 / DHGT"),
            _record(_grid_payload(32.5, 31.5, -1.0), "LAT1 / LAT2 / DLAT"),
            _record(_grid_payload(-100.0, -98.0, 1.0), "LON1 / LON2 / DLON"),
            _record(f"{-1:6d}", "EXPONENT"),
            _record("", "END OF HEADER"),
        )
    )
    return header + _map(1, 0, 100) + _record("", "END OF FILE")


class _BarrierAggregator:
    def __init__(self, delegate, barrier: Barrier) -> None:
        self._delegate = delegate
        self._barrier = barrier
        self._thread_state = local()

    def aggregate(self, **kwargs):
        if not getattr(self._thread_state, "waited", False):
            self._thread_state.waited = True
            self._barrier.wait(timeout=5)
        return self._delegate.aggregate(**kwargs)


class _DistinctClock:
    def __init__(self) -> None:
        self._lock = Lock()
        self._offset_seconds = 0

    def __call__(self) -> datetime:
        with self._lock:
            self._offset_seconds += 1
            return RUN_AT + timedelta(seconds=self._offset_seconds)


class _BlockingCommitUnitOfWork(SQLiteUnitOfWork):
    commit_started: Event
    release_commit: Event

    def commit(self) -> None:
        self.commit_started.set()
        if not self.release_commit.wait(timeout=5):
            raise TimeoutError("test did not release aggregation commit")
        super().commit()


class VerticalSliceWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.database = root / "ophanim.sqlite3"
        self.artifact_directory = root / "artifacts"
        self.source = root / "two-epoch.inx"
        self.source.write_text(_two_epoch_ionex(), encoding="ascii")

        self.processing = ProcessingVersion(
            processing_version_id="processing-v1",
            code_revision="test-revision",
            configuration_hash="processing-config",
            parser_version=IONEXV1Parser.parser_version,
            created_at=ORIGIN - timedelta(days=1),
        )
        self.bounds = GeographicBounds(
            south_latitude_degrees=31.5,
            west_longitude_degrees=-100.0,
            north_latitude_degrees=32.5,
            east_longitude_degrees=-98.0,
        )
        self.region = RegionVersion(
            region_version_id="central-texas-v1",
            name="Central Texas test grid",
            boundary_ref="central-texas-test-bounds",
            boundary_checksum_sha256=boundary_checksum_sha256(self.bounds),
            created_at=ORIGIN - timedelta(days=1),
        )
        self.model = ModelVersion(
            model_version_id="persistence-v1",
            model_type="persistence",
            configuration_hash="persistence-config",
            created_at=ORIGIN - timedelta(days=1),
        )
        self.detector = DetectorVersion(
            detector_version_id="residual-z-v1",
            scoring_method="residual_zscore",
            threshold=3.0,
            configuration_hash="residual-z-config",
            created_at=ORIGIN - timedelta(days=1),
            calibration_data_version="calibration-v1",
            calibration_residual_mean_tecu=0.0,
            calibration_residual_standard_deviation_tecu=2.0,
            calibration_sample_count=100,
        )
        self.application = create_sqlite_application(
            database=self.database,
            artifact_directory=self.artifact_directory,
            region_bounds={
                self.region.boundary_ref: self.bounds
            },
            aggregation_policy=AggregationPolicy(expected_cell_count=6),
            detector_version_id=self.detector.detector_version_id,
            clock=lambda: RUN_AT,
        )
        self.ingestion_request = IngestionRequest(
            provider="test-provider",
            product="test-ionex",
            source_uri=str(self.source),
            revision="r1",
        )

    def _request(self) -> VerticalSliceRequest:
        return VerticalSliceRequest(
            ingestion=self.ingestion_request,
            region=self.region,
            processing_version=self.processing,
            model_version=self.model,
            detector_version=self.detector,
            forecast_origin=ORIGIN,
            target_metric=TargetMetric.MEDIAN_VTEC,
            mode=ForecastMode.HINDCAST,
        )

    def test_one_file_runs_through_ingestion_forecast_and_decision(self) -> None:
        result = self.application.vertical_slice.run(self._request())

        self.assertEqual(result.ingestion.observation_count, 12)
        self.assertEqual(result.aggregation.epochs_seen, 2)
        self.assertEqual(result.aggregation.observations_created, 2)
        self.assertEqual(result.forecast.forecast.predicted_vtec_tecu, 10.0)
        self.assertIs(result.forecast.forecast.mode, ForecastMode.HINDCAST)
        self.assertIs(result.forecast.forecast.status, ForecastStatus.SCORED)
        self.assertEqual(result.reconciliation.scored, 1)
        self.assertEqual(len(result.decisions), 1)
        decision = result.decisions[0]
        self.assertEqual(decision.residual_tecu, 10.0)
        self.assertEqual(decision.anomaly_score, 5.0)
        self.assertTrue(decision.is_forecast_anomaly)
        self.assertIs(
            decision.disturbance_assessment,
            DisturbanceAssessment.CANDIDATE,
        )
        self.assertEqual(decision.scored_at, RUN_AT)

        with SQLiteUnitOfWork(self.database) as unit_of_work:
            stored = unit_of_work.forecasts.get(result.forecast.forecast.forecast_id)
            self.assertIsNotNone(stored)
            self.assertIs(stored.status, ForecastStatus.SCORED)

    def test_full_retry_reuses_every_durable_result(self) -> None:
        first = self.application.vertical_slice.run(self._request())
        second = self.application.vertical_slice.run(self._request())

        self.assertTrue(second.ingestion.already_present)
        self.assertEqual(second.aggregation.observations_created, 0)
        self.assertEqual(second.aggregation.observations_reused, 2)
        self.assertTrue(second.forecast.already_present)
        self.assertEqual(
            second.forecast.forecast.forecast_id,
            first.forecast.forecast.forecast_id,
        )
        self.assertEqual(second.reconciliation.considered, 0)
        self.assertEqual(second.decisions, first.decisions)

    def test_concurrent_aggregation_rechecks_idempotency_under_write_lock(
        self,
    ) -> None:
        ingestion = self.application.ingester.ingest(self.ingestion_request)
        delegate = MeanMedianRegionalAggregator(
            selection_strategy=BoundsRegionSelector(
                {self.region.boundary_ref: self.bounds}
            ),
            policy=AggregationPolicy(expected_cell_count=6),
        )
        workflow = RegionalAggregationWorkflow(
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            aggregator=_BarrierAggregator(delegate, Barrier(2)),
            clock=_DistinctClock(),
        )

        def aggregate():
            return workflow.aggregate_artifact(
                artifact_id=ingestion.artifact.artifact_id,
                region=self.region,
                processing_version=self.processing,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [future.result(timeout=10) for future in (
                executor.submit(aggregate),
                executor.submit(aggregate),
            )]

        self.assertEqual(
            sorted(result.observations_created for result in results),
            [0, 2],
        )
        self.assertEqual(
            sorted(result.observations_reused for result in results),
            [0, 2],
        )
        with SQLiteUnitOfWork(self.database) as unit_of_work:
            stored = unit_of_work.regional_observations.for_artifact(
                ingestion.artifact.artifact_id
            )
        self.assertEqual(len(stored), 2)

    def test_live_issuance_is_stamped_after_concurrent_publication_commit(
        self,
    ) -> None:
        self.source.write_text(_one_epoch_ionex(), encoding="ascii")
        early_application = create_sqlite_application(
            database=self.database,
            artifact_directory=self.artifact_directory,
            region_bounds={self.region.boundary_ref: self.bounds},
            aggregation_policy=AggregationPolicy(expected_cell_count=6),
            detector_version_id=self.detector.detector_version_id,
            clock=lambda: ORIGIN + timedelta(minutes=20),
        )
        ingestion = early_application.ingester.ingest(self.ingestion_request)
        commit_started = Event()
        release_commit = Event()
        issue_entered = Event()
        _BlockingCommitUnitOfWork.commit_started = commit_started
        _BlockingCommitUnitOfWork.release_commit = release_commit

        aggregation = RegionalAggregationWorkflow(
            unit_of_work_factory=partial(
                _BlockingCommitUnitOfWork,
                self.database,
            ),
            aggregator=MeanMedianRegionalAggregator(
                selection_strategy=BoundsRegionSelector(
                    {self.region.boundary_ref: self.bounds}
                ),
                policy=AggregationPolicy(expected_cell_count=6),
            ),
            clock=lambda: ORIGIN + timedelta(minutes=30),
        )

        def issue_unit_of_work():
            issue_entered.set()
            return SQLiteUnitOfWork(self.database)

        issued_at = ORIGIN + timedelta(minutes=40)

        def issuance_clock() -> datetime:
            if not release_commit.is_set():
                raise AssertionError("issuance was stamped before publication commit")
            return issued_at

        issuing = ForecastIssuingWorkflow(
            unit_of_work_factory=issue_unit_of_work,
            model_runner=PersistenceModelRunner(),
            clock=issuance_clock,
        )
        request = ForecastRequest(
            origin_at=ORIGIN,
            valid_at=VALID_AT,
            source_provider=self.ingestion_request.provider,
            source_product=self.ingestion_request.product,
            region_version_id=self.region.region_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            mode=ForecastMode.LIVE,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            aggregation_future = executor.submit(
                aggregation.aggregate_artifact,
                artifact_id=ingestion.artifact.artifact_id,
                region=self.region,
                processing_version=self.processing,
            )
            if not commit_started.wait(timeout=5):
                error = (
                    aggregation_future.exception(timeout=0)
                    if aggregation_future.done()
                    else "aggregation remained blocked"
                )
                self.fail(f"aggregation did not reach commit: {error}")
            issuance_future = executor.submit(
                issuing.issue,
                request=request,
                model_version=self.model,
                processing_version=self.processing,
            )
            self.assertTrue(issue_entered.wait(timeout=5))
            release_commit.set()
            aggregation_future.result(timeout=10)
            result = issuance_future.result(timeout=10)

        self.assertEqual(result.forecast.issued_at, issued_at)
        self.assertEqual(result.forecast.predicted_vtec_tecu, 10.0)

    def test_live_forecast_cannot_use_an_artifact_ingested_in_the_future(self) -> None:
        self.application.vertical_slice.run(self._request())
        issuing = ForecastIssuingWorkflow(
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            model_runner=PersistenceModelRunner(),
            clock=lambda: ORIGIN + timedelta(minutes=30),
        )
        live_request = ForecastRequest(
            origin_at=ORIGIN,
            valid_at=VALID_AT,
            source_provider=self.ingestion_request.provider,
            source_product=self.ingestion_request.product,
            region_version_id=self.region.region_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            mode=ForecastMode.LIVE,
        )

        with self.assertRaises(InsufficientHistoryError):
            issuing.issue(
                request=live_request,
                model_version=self.model,
                processing_version=self.processing,
            )
        with self.assertRaisesRegex(ValueError, "workflow-owned"):
            issuing.issue(
                request=replace(
                    live_request,
                    issued_at=ORIGIN + timedelta(minutes=30),
                ),
                model_version=self.model,
                processing_version=self.processing,
            )

    def test_live_forecast_finishing_at_valid_time_is_not_persisted(self) -> None:
        self.source.write_text(_one_epoch_ionex(), encoding="ascii")
        early_application = create_sqlite_application(
            database=self.database,
            artifact_directory=self.artifact_directory,
            region_bounds={self.region.boundary_ref: self.bounds},
            aggregation_policy=AggregationPolicy(expected_cell_count=6),
            detector_version_id=self.detector.detector_version_id,
            clock=lambda: ORIGIN + timedelta(minutes=20),
        )
        ingestion = early_application.ingester.ingest(self.ingestion_request)
        early_application.aggregation.aggregate_artifact(
            artifact_id=ingestion.artifact.artifact_id,
            region=self.region,
            processing_version=self.processing,
        )
        instants = iter((ORIGIN + timedelta(minutes=40), VALID_AT))
        issuing = ForecastIssuingWorkflow(
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            model_runner=PersistenceModelRunner(),
            clock=lambda: next(instants),
        )
        request = ForecastRequest(
            origin_at=ORIGIN,
            valid_at=VALID_AT,
            source_provider=self.ingestion_request.provider,
            source_product=self.ingestion_request.product,
            region_version_id=self.region.region_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            mode=ForecastMode.LIVE,
        )

        with self.assertRaisesRegex(ForecastingError, "did not finish"):
            issuing.issue(
                request=request,
                model_version=self.model,
                processing_version=self.processing,
            )
        with SQLiteUnitOfWork(self.database) as unit_of_work:
            self.assertEqual(
                unit_of_work.forecasts.list_pending(
                    valid_at_or_before=VALID_AT + timedelta(days=1)
                ),
                (),
            )

    def test_live_vertical_slice_issues_after_fresh_data_is_available(self) -> None:
        self.source.write_text(_one_epoch_ionex(), encoding="ascii")
        instants = iter(
            ORIGIN + timedelta(minutes=minutes)
            for minutes in (10, 20, 30, 40, 50, 60, 70)
        )
        application = create_sqlite_application(
            database=self.database,
            artifact_directory=self.artifact_directory,
            region_bounds={self.region.boundary_ref: self.bounds},
            aggregation_policy=AggregationPolicy(expected_cell_count=6),
            detector_version_id=self.detector.detector_version_id,
            clock=lambda: next(instants),
        )
        request = replace(self._request(), mode=ForecastMode.LIVE)

        result = application.vertical_slice.run(request)

        self.assertEqual(
            result.forecast.forecast.issued_at,
            ORIGIN + timedelta(minutes=50),
        )
        self.assertEqual(result.forecast.forecast.predicted_vtec_tecu, 10.0)
        self.assertIs(result.forecast.forecast.status, ForecastStatus.PENDING)
        self.assertEqual(result.reconciliation.scored, 0)

    def test_identical_bytes_keep_distinct_acquisition_metadata(self) -> None:
        first = self.application.ingester.ingest(self.ingestion_request)
        second = self.application.ingester.ingest(
            IngestionRequest(
                provider="another-provider",
                product=self.ingestion_request.product,
                source_uri=self.ingestion_request.source_uri,
                revision=self.ingestion_request.revision,
            )
        )

        self.assertNotEqual(first.artifact.artifact_id, second.artifact.artifact_id)
        self.assertEqual(first.artifact.storage_ref, second.artifact.storage_ref)
        stored_files = [
            path for path in self.artifact_directory.rglob("*") if path.is_file()
        ]
        self.assertEqual(len(stored_files), 1)

    def test_retry_cannot_silently_change_an_equivalent_forecast(self) -> None:
        self.application.vertical_slice.run(self._request())

        class AlteredRunner:
            def run(inner_self, **kwargs):
                forecast = PersistenceModelRunner().run(**kwargs)
                return replace(
                    forecast,
                    predicted_vtec_tecu=forecast.predicted_vtec_tecu + 1.0,
                )

        issuing = ForecastIssuingWorkflow(
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            model_runner=AlteredRunner(),
        )
        request = ForecastRequest(
            origin_at=ORIGIN,
            issued_at=RUN_AT,
            valid_at=VALID_AT,
            source_provider=self.ingestion_request.provider,
            source_product=self.ingestion_request.product,
            region_version_id=self.region.region_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            mode=ForecastMode.HINDCAST,
        )

        with self.assertRaises(ForecastConflictError):
            issuing.issue(
                request=request,
                model_version=self.model,
                processing_version=self.processing,
            )


if __name__ == "__main__":
    unittest.main()
