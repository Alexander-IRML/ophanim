"""Tests for pending forecast lifecycle resolution."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Barrier

from ophanim.detection import ResidualZScoreDecisionEngine
from ophanim.domain import (
    DetectorVersion,
    Forecast,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    SourceArtifact,
    TargetMetric,
)
from ophanim.reconciliation import (
    DefaultPendingForecastReconciler,
    ReconciliationSummary,
)
from ophanim.sqlite import SQLiteUnitOfWork


UTC = timezone.utc
T0 = datetime(2026, 1, 2, 0, tzinfo=UTC)


class ReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.database = Path(self._temporary_directory.name) / "ophanim.sqlite3"
        self.uow_factory = partial(SQLiteUnitOfWork, self.database)

        self.artifact = SourceArtifact(
            artifact_id="artifact-v1",
            provider="test",
            product="test",
            parser_version="test-parser/1",
            revision_priority=0,
            checksum_sha256="a" * 64,
            storage_ref="artifact.inx",
            ingested_at=T0,
        )
        self.processing = ProcessingVersion(
            processing_version_id="processing-v1",
            code_revision="test",
            configuration_hash="processing-config",
            parser_version=self.artifact.parser_version,
            created_at=T0 - timedelta(days=1),
        )
        self.region = RegionVersion(
            region_version_id="region-v1",
            name="Test region",
            boundary_ref="test-bounds",
            boundary_checksum_sha256="c" * 64,
            created_at=T0 - timedelta(days=1),
        )
        self.model = ModelVersion(
            model_version_id="persistence-v1",
            model_type="persistence",
            configuration_hash="model-config",
            created_at=T0 - timedelta(days=1),
        )
        self.detector = DetectorVersion(
            detector_version_id="detector-v1",
            scoring_method="residual_zscore",
            threshold=3.0,
            configuration_hash="detector-config",
            created_at=T0 - timedelta(days=1),
            calibration_data_version="calibration-v1",
            calibration_residual_mean_tecu=0.0,
            calibration_residual_standard_deviation_tecu=2.0,
            calibration_sample_count=100,
        )
        self.history = RegionalObservation(
            observation_id="history-v1",
            artifact_id=self.artifact.artifact_id,
            processing_version_id=self.processing.processing_version_id,
            region_version_id=self.region.region_version_id,
            observed_at=T0,
            produced_at=T0,
            mean_vtec_tecu=10.0,
            median_vtec_tecu=10.0,
            cell_count=6,
            coverage_fraction=1.0,
        )
        self.actual = replace(
            self.history,
            observation_id="actual-v1",
            observed_at=T0 + timedelta(hours=2),
            produced_at=T0 + timedelta(hours=2, minutes=5),
            mean_vtec_tecu=20.0,
            median_vtec_tecu=20.0,
        )
        self.forecast = Forecast(
            forecast_id="forecast-v1",
            origin_at=T0,
            issued_at=T0,
            valid_at=T0 + timedelta(hours=2),
            source_provider=self.artifact.provider,
            source_product=self.artifact.product,
            region_version_id=self.region.region_version_id,
            model_version_id=self.model.model_version_id,
            processing_version_id=self.processing.processing_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            predicted_vtec_tecu=10.0,
            input_observation_ids=(self.history.observation_id,),
        )

    def _store(self, *, actual: RegionalObservation | None = None, detector=None) -> None:
        with self.uow_factory() as unit_of_work:
            unit_of_work.artifacts.add(self.artifact)
            unit_of_work.processing_versions.register(self.processing)
            unit_of_work.regions.register(self.region)
            unit_of_work.models.register(self.model)
            unit_of_work.detectors.register(detector or self.detector)
            unit_of_work.regional_observations.add(self.history)
            if actual is not None:
                unit_of_work.regional_observations.add(actual)
            unit_of_work.forecasts.add(self.forecast)
            unit_of_work.commit()

    def _reconciler(self, *, now: datetime, grace_hours: float = 6.0):
        return DefaultPendingForecastReconciler(
            unit_of_work_factory=self.uow_factory,
            decision_engine=ResidualZScoreDecisionEngine(),
            detector_version_id=self.detector.detector_version_id,
            observation_grace_period_seconds=grace_hours * 60 * 60,
            clock=lambda: now,
        )

    def test_reconciler_configuration_rejects_invalid_grace_and_detector(self) -> None:
        for invalid_grace in (float("nan"), float("inf"), -1.0, True, "six"):
            with self.subTest(invalid_grace=invalid_grace):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    DefaultPendingForecastReconciler(
                        unit_of_work_factory=self.uow_factory,
                        decision_engine=ResidualZScoreDecisionEngine(),
                        detector_version_id=self.detector.detector_version_id,
                        observation_grace_period_seconds=invalid_grace,
                    )
        with self.assertRaisesRegex(ValueError, "detector_version_id"):
            DefaultPendingForecastReconciler(
                unit_of_work_factory=self.uow_factory,
                decision_engine=ResidualZScoreDecisionEngine(),
                detector_version_id=" ",
            )

    def test_low_quality_actual_becomes_insufficient_data(self) -> None:
        self._store(actual=replace(self.actual, coverage_fraction=0.25))

        summary = self._reconciler(
            now=T0 + timedelta(hours=3),
            grace_hours=0,
        ).reconcile(valid_at_or_before=self.forecast.valid_at)

        self.assertEqual(summary.insufficient_data, 1)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.INSUFFICIENT_DATA)
            self.assertEqual(unit_of_work.decisions.for_forecast(stored.forecast_id), ())

    def test_better_revision_can_replace_low_quality_actual_during_grace(self) -> None:
        self._store(actual=replace(self.actual, coverage_fraction=0.25))

        first = self._reconciler(now=T0 + timedelta(hours=3)).reconcile(
            valid_at_or_before=self.forecast.valid_at
        )
        self.assertEqual(first.insufficient_data, 0)
        with self.uow_factory() as unit_of_work:
            pending = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(pending.status, ForecastStatus.PENDING)

        revised_artifact = replace(
            self.artifact,
            artifact_id="artifact-revised",
            checksum_sha256="b" * 64,
            storage_ref="artifact-revised.inx",
            ingested_at=T0 + timedelta(hours=3, minutes=5),
            revision="r2",
        )
        revised_actual = replace(
            self.actual,
            observation_id="actual-revised",
            artifact_id=revised_artifact.artifact_id,
            produced_at=revised_artifact.ingested_at,
        )
        with self.uow_factory() as unit_of_work:
            unit_of_work.artifacts.add(revised_artifact)
            unit_of_work.regional_observations.add(revised_actual)
            unit_of_work.commit()

        second = self._reconciler(
            now=T0 + timedelta(hours=3, minutes=10)
        ).reconcile(valid_at_or_before=self.forecast.valid_at)
        self.assertEqual(second.scored, 1)
        with self.uow_factory() as unit_of_work:
            scored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(scored.status, ForecastStatus.SCORED)
            decisions = unit_of_work.decisions.for_forecast(scored.forecast_id)
            self.assertEqual(decisions[0].observation_id, revised_actual.observation_id)

    def test_operational_or_configuration_error_remains_retryable(self) -> None:
        invalid_detector = replace(
            self.detector,
            calibration_residual_standard_deviation_tecu=None,
        )
        self._store(actual=self.actual, detector=invalid_detector)

        with self.assertLogs("ophanim.reconciliation", level="ERROR"):
            summary = self._reconciler(now=T0 + timedelta(hours=3)).reconcile(
                valid_at_or_before=self.forecast.valid_at
            )

        self.assertEqual(summary.retryable_errors, 1)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.PENDING)

    def test_targeted_reconciliation_scores_only_the_requested_forecast(
        self,
    ) -> None:
        self._store(actual=self.actual)
        other_forecast = replace(
            self.forecast,
            forecast_id="forecast-other",
            target_metric=TargetMetric.MEAN_VTEC,
        )
        with self.uow_factory() as unit_of_work:
            unit_of_work.forecasts.add(other_forecast)
            unit_of_work.commit()

        summary = self._reconciler(
            now=T0 + timedelta(hours=3)
        ).reconcile_forecast(
            forecast_id=self.forecast.forecast_id,
            valid_at_or_before=self.forecast.valid_at,
        )

        self.assertEqual(
            summary,
            ReconciliationSummary(
                considered=1,
                scored=1,
                insufficient_data=0,
                retryable_errors=0,
            ),
        )
        with self.uow_factory() as unit_of_work:
            requested = unit_of_work.forecasts.get(self.forecast.forecast_id)
            untouched = unit_of_work.forecasts.get(other_forecast.forecast_id)
            self.assertIs(requested.status, ForecastStatus.SCORED)
            self.assertIs(untouched.status, ForecastStatus.PENDING)
            decisions = unit_of_work.decisions.for_forecast(
                self.forecast.forecast_id
            )
            self.assertEqual(len(decisions), 1)
            self.assertEqual(
                decisions[0].detector_version_id,
                self.detector.detector_version_id,
            )
            self.assertEqual(
                unit_of_work.decisions.for_forecast(other_forecast.forecast_id),
                (),
            )

    def test_targeted_reconciliation_obeys_cutoff_and_pending_status(self) -> None:
        self._store(actual=self.actual)
        reconciler = self._reconciler(now=T0 + timedelta(hours=3))

        before_validity = reconciler.reconcile_forecast(
            forecast_id=self.forecast.forecast_id,
            valid_at_or_before=self.forecast.valid_at - timedelta(microseconds=1),
        )
        scored = reconciler.reconcile_forecast(
            forecast_id=self.forecast.forecast_id,
            valid_at_or_before=self.forecast.valid_at,
        )
        repeated = reconciler.reconcile_forecast(
            forecast_id=self.forecast.forecast_id,
            valid_at_or_before=self.forecast.valid_at,
        )

        empty = ReconciliationSummary(0, 0, 0, 0)
        self.assertEqual(before_validity, empty)
        self.assertEqual(scored, ReconciliationSummary(1, 1, 0, 0))
        self.assertEqual(repeated, empty)

    def test_targeted_reconciliation_reports_retryable_failure(self) -> None:
        invalid_detector = replace(
            self.detector,
            calibration_residual_standard_deviation_tecu=None,
        )
        self._store(actual=self.actual, detector=invalid_detector)

        with self.assertLogs("ophanim.reconciliation", level="ERROR"):
            summary = self._reconciler(
                now=T0 + timedelta(hours=3)
            ).reconcile_forecast(
                forecast_id=self.forecast.forecast_id,
                valid_at_or_before=self.forecast.valid_at,
            )

        self.assertEqual(summary, ReconciliationSummary(1, 0, 0, 1))
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.PENDING)

    def test_targeted_reconciliation_rejects_unknown_forecast(self) -> None:
        self._store()

        with self.assertRaisesRegex(LookupError, "forecast does not exist"):
            self._reconciler(
                now=T0 + timedelta(hours=3)
            ).reconcile_forecast(
                forecast_id="forecast-unknown",
                valid_at_or_before=self.forecast.valid_at,
            )

    def test_scored_forecast_can_supersede_a_decision_for_revised_actual(self) -> None:
        self._store(actual=self.actual)
        reconciler = self._reconciler(now=T0 + timedelta(hours=3))
        first_summary = reconciler.reconcile(
            valid_at_or_before=self.forecast.valid_at
        )
        self.assertEqual(first_summary.scored, 1)
        with self.uow_factory() as unit_of_work:
            first_decision = unit_of_work.decisions.for_forecast(
                self.forecast.forecast_id
            )[0]

        revised_artifact = replace(
            self.artifact,
            artifact_id="artifact-post-score-revision",
            checksum_sha256="e" * 64,
            storage_ref="artifact-post-score-revision.inx",
            ingested_at=T0 + timedelta(hours=4),
            revision="r2",
        )
        revised_actual = replace(
            self.actual,
            observation_id="actual-post-score-revision",
            artifact_id=revised_artifact.artifact_id,
            produced_at=revised_artifact.ingested_at,
            mean_vtec_tecu=11.0,
            median_vtec_tecu=11.0,
        )
        with self.uow_factory() as unit_of_work:
            unit_of_work.artifacts.add(revised_artifact)
            unit_of_work.regional_observations.add(revised_actual)
            unit_of_work.commit()

        revised_decision = self._reconciler(
            now=T0 + timedelta(hours=4, minutes=1)
        ).reassess_scored(forecast_id=self.forecast.forecast_id)

        self.assertIsNotNone(revised_decision)
        self.assertEqual(
            revised_decision.supersedes_decision_id,
            first_decision.decision_id,
        )
        self.assertEqual(revised_decision.observation_id, revised_actual.observation_id)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.SCORED)
            self.assertEqual(
                len(unit_of_work.decisions.for_forecast(stored.forecast_id)),
                2,
            )
            query = {
                "valid_at_or_after": self.forecast.valid_at - timedelta(minutes=1),
                "valid_before": self.forecast.valid_at + timedelta(minutes=1),
            }
            self.assertEqual(
                unit_of_work.decisions.search(**query),
                (revised_decision,),
            )
            self.assertEqual(
                unit_of_work.decisions.search(
                    **query,
                    include_superseded=True,
                ),
                (first_decision, revised_decision),
            )
            self.assertEqual(
                unit_of_work.decisions.search(
                    **query,
                    is_forecast_anomaly=True,
                ),
                (),
            )

    def test_concurrent_sweeps_count_only_the_winning_transition(self) -> None:
        self._store(actual=self.actual)
        barrier = Barrier(2)

        class BarrierReconciler(DefaultPendingForecastReconciler):
            def _reconcile_one(inner_self, **kwargs):
                barrier.wait(timeout=5)
                return super()._reconcile_one(**kwargs)

        def reconcile():
            return BarrierReconciler(
                unit_of_work_factory=self.uow_factory,
                decision_engine=ResidualZScoreDecisionEngine(),
                detector_version_id=self.detector.detector_version_id,
                clock=lambda: T0 + timedelta(hours=3),
            ).reconcile(valid_at_or_before=self.forecast.valid_at)

        with ThreadPoolExecutor(max_workers=2) as executor:
            summaries = [
                future.result(timeout=10)
                for future in (
                    executor.submit(reconcile),
                    executor.submit(reconcile),
                )
            ]

        self.assertEqual(sorted(summary.scored for summary in summaries), [0, 1])
        with self.uow_factory() as unit_of_work:
            decisions = unit_of_work.decisions.for_forecast(
                self.forecast.forecast_id
            )
        self.assertEqual(len(decisions), 1)

    def test_reconciliation_cannot_use_a_revision_unavailable_at_score_time(
        self,
    ) -> None:
        self._store(actual=self.actual)
        future_artifact = replace(
            self.artifact,
            artifact_id="artifact-future-revision",
            revision_priority=10,
            checksum_sha256="f" * 64,
            storage_ref="artifact-future-revision.inx",
            ingested_at=T0 + timedelta(hours=4),
            revision="final",
        )
        future_actual = replace(
            self.actual,
            observation_id="actual-future-revision",
            artifact_id=future_artifact.artifact_id,
            produced_at=future_artifact.ingested_at,
            median_vtec_tecu=99.0,
        )
        with self.uow_factory() as unit_of_work:
            unit_of_work.artifacts.add(future_artifact)
            unit_of_work.regional_observations.add(future_actual)
            unit_of_work.commit()

        summary = self._reconciler(now=T0 + timedelta(hours=3)).reconcile(
            valid_at_or_before=self.forecast.valid_at
        )

        self.assertEqual(summary.scored, 1)
        with self.uow_factory() as unit_of_work:
            decisions = unit_of_work.decisions.for_forecast(
                self.forecast.forecast_id
            )
        self.assertEqual(decisions[0].observation_id, self.actual.observation_id)

    def test_missing_actual_waits_for_grace_then_becomes_insufficient(self) -> None:
        self._store()

        first = self._reconciler(
            now=T0 + timedelta(hours=3),
            grace_hours=2,
        ).reconcile(valid_at_or_before=self.forecast.valid_at)
        self.assertEqual(first.insufficient_data, 0)

        second = self._reconciler(
            now=T0 + timedelta(hours=5),
            grace_hours=2,
        ).reconcile(valid_at_or_before=self.forecast.valid_at)
        self.assertEqual(second.insufficient_data, 1)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.INSUFFICIENT_DATA)

    def test_late_actual_can_score_an_insufficient_forecast(self) -> None:
        self._store()
        terminal_reconciler = self._reconciler(
            now=T0 + timedelta(hours=5),
            grace_hours=2,
        )
        terminal_reconciler.reconcile(valid_at_or_before=self.forecast.valid_at)

        late_artifact = replace(
            self.artifact,
            artifact_id="artifact-late-actual",
            checksum_sha256="d" * 64,
            storage_ref="artifact-late-actual.inx",
            ingested_at=T0 + timedelta(hours=5, minutes=5),
            revision="late",
        )
        late_actual = replace(
            self.actual,
            observation_id="actual-late",
            artifact_id=late_artifact.artifact_id,
            produced_at=late_artifact.ingested_at,
        )
        with self.uow_factory() as unit_of_work:
            unit_of_work.artifacts.add(late_artifact)
            unit_of_work.regional_observations.add(late_actual)
            unit_of_work.commit()

        decision = self._reconciler(
            now=T0 + timedelta(hours=5, minutes=10),
        ).retry_insufficient(forecast_id=self.forecast.forecast_id)
        repeated = self._reconciler(
            now=T0 + timedelta(hours=5, minutes=11),
        ).retry_insufficient(forecast_id=self.forecast.forecast_id)

        self.assertIsNotNone(decision)
        self.assertEqual(repeated, decision)
        self.assertEqual(decision.observation_id, late_actual.observation_id)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.SCORED)
            self.assertEqual(
                unit_of_work.decisions.for_forecast(self.forecast.forecast_id),
                (decision,),
            )

    def test_retry_without_a_usable_actual_remains_insufficient(self) -> None:
        self._store()
        reconciler = self._reconciler(
            now=T0 + timedelta(hours=5),
            grace_hours=2,
        )
        reconciler.reconcile(valid_at_or_before=self.forecast.valid_at)

        decision = self._reconciler(
            now=T0 + timedelta(hours=6),
        ).retry_insufficient(forecast_id=self.forecast.forecast_id)

        self.assertIsNone(decision)
        with self.uow_factory() as unit_of_work:
            stored = unit_of_work.forecasts.get(self.forecast.forecast_id)
            self.assertIs(stored.status, ForecastStatus.INSUFFICIENT_DATA)
            self.assertEqual(
                unit_of_work.decisions.for_forecast(self.forecast.forecast_id),
                (),
            )


if __name__ == "__main__":
    unittest.main()
