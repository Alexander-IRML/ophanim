"""Behavior tests for the stdlib SQLite persistence adapter."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    DisturbanceAssessment,
    Forecast,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    SourceArtifact,
    TargetMetric,
    TECObservation,
)
from ophanim.sqlite import SCHEMA_VERSION, SQLiteUnitOfWork


UTC = timezone.utc
T0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)


class SQLitePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.database = Path(self._temporary_directory.name) / "ophanim.sqlite3"

        self.artifact = SourceArtifact(
            artifact_id="artifact-1",
            provider="cddis",
            product="ionex-final",
            parser_version="test-parser/1",
            revision_priority=10,
            checksum_sha256="a" * 64,
            storage_ref="artifacts/artifact-1.ionex",
            ingested_at=T0 - timedelta(minutes=5),
            revision="r1",
            source_uri="https://example.test/artifact-1.ionex",
        )
        self.processing = ProcessingVersion(
            processing_version_id="processing-1",
            code_revision="git:abc123",
            configuration_hash="processing-config-hash",
            parser_version=self.artifact.parser_version,
            created_at=T0,
        )
        self.region = RegionVersion(
            region_version_id="central-texas-v1",
            name="Central Texas",
            boundary_ref="regions/central-texas-v1.geojson",
            boundary_checksum_sha256="c" * 64,
            created_at=T0,
        )
        self.tec = TECObservation(
            artifact_id=self.artifact.artifact_id,
            observed_at=T0,
            latitude_degrees=30.25,
            longitude_degrees=-97.75,
            vtec_tecu=12.5,
            quality_flags=("interpolated", "edge-cell"),
        )
        self.observation = RegionalObservation(
            observation_id="observation-1",
            artifact_id=self.artifact.artifact_id,
            processing_version_id=self.processing.processing_version_id,
            region_version_id=self.region.region_version_id,
            observed_at=T0,
            produced_at=T0,
            mean_vtec_tecu=12.8,
            median_vtec_tecu=12.5,
            cell_count=42,
            coverage_fraction=0.875,
            quality_flags=("partial-coverage",),
        )
        self.model = ModelVersion(
            model_version_id="persistence-v1",
            model_type="persistence",
            configuration_hash="model-config-hash",
            created_at=T0,
            artifact_ref="models/persistence-v1.json",
            artifact_checksum_sha256="d" * 64,
            training_data_version="observations-2025",
        )
        self.detector = DetectorVersion(
            detector_version_id="residual-z-v1",
            scoring_method="residual_z_score",
            threshold=2.5,
            configuration_hash="detector-config-hash",
            created_at=T0,
            calibration_data_version="residuals-2025",
            calibration_residual_mean_tecu=0.15,
            calibration_residual_standard_deviation_tecu=1.25,
            calibration_sample_count=8760,
        )
        self.forecast = Forecast(
            forecast_id="forecast-1",
            origin_at=T0,
            issued_at=T0,
            valid_at=T0 + timedelta(hours=2),
            source_provider=self.artifact.provider,
            source_product=self.artifact.product,
            region_version_id=self.region.region_version_id,
            model_version_id=self.model.model_version_id,
            processing_version_id=self.processing.processing_version_id,
            target_metric=TargetMetric.MEDIAN_VTEC,
            predicted_vtec_tecu=12.5,
            input_observation_ids=(self.observation.observation_id,),
        )
        self.actual = replace(
            self.observation,
            observation_id="actual-1",
            observed_at=self.forecast.valid_at,
            produced_at=self.forecast.valid_at + timedelta(seconds=30),
            mean_vtec_tecu=16.8,
            median_vtec_tecu=16.5,
        )
        self.decision = AnomalyDecision(
            decision_id="decision-1",
            forecast_id=self.forecast.forecast_id,
            observation_id=self.actual.observation_id,
            detector_version_id=self.detector.detector_version_id,
            scored_at=T0 + timedelta(hours=2, minutes=1),
            residual_tecu=4.0,
            anomaly_score=3.2,
            threshold=2.5,
            is_forecast_anomaly=True,
            disturbance_assessment=DisturbanceAssessment.CANDIDATE,
            assessment_source="residual-rule",
        )

    def _add_dependencies(self, uow: SQLiteUnitOfWork) -> None:
        uow.artifacts.add(self.artifact)
        uow.processing_versions.register(self.processing)
        uow.regions.register(self.region)
        uow.models.register(self.model)
        uow.detectors.register(self.detector)

    def _add_through_forecast(self, uow: SQLiteUnitOfWork) -> None:
        self._add_dependencies(uow)
        uow.tec_observations.add_many((self.tec,))
        uow.regional_observations.add(self.observation)
        uow.forecasts.add(self.forecast)

    def test_round_trips_every_domain_record_enum_list_and_flag(self) -> None:
        prior_observation = replace(
            self.observation,
            observation_id="observation-prior",
            observed_at=T0 - timedelta(hours=1),
            mean_vtec_tecu=11.9,
            median_vtec_tecu=11.7,
        )
        forecast = replace(
            self.forecast,
            input_observation_ids=(
                prior_observation.observation_id,
                self.observation.observation_id,
            ),
        )
        confirming_detector = replace(
            self.detector,
            detector_version_id="external-confirmation-v1",
            scoring_method="external_confirmation",
            configuration_hash="confirming-detector-config-hash",
        )
        confirming_decision = replace(
            self.decision,
            decision_id="decision-2",
            detector_version_id=confirming_detector.detector_version_id,
            scored_at=self.decision.scored_at + timedelta(minutes=1),
            disturbance_assessment=DisturbanceAssessment.CONFIRMED,
            assessment_source="external-index",
            supersedes_decision_id=self.decision.decision_id,
        )
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_dependencies(uow)
            uow.detectors.register(confirming_detector)
            uow.tec_observations.add_many((self.tec,))
            uow.regional_observations.add(prior_observation)
            uow.regional_observations.add(self.observation)
            uow.regional_observations.add(self.actual)
            uow.forecasts.add(forecast)
            uow.decisions.add(self.decision)
            uow.decisions.add(confirming_decision)
            uow.forecasts.replace(
                replace(forecast, status=ForecastStatus.SCORED)
            )
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(uow.artifacts.get(self.artifact.artifact_id), self.artifact)
            self.assertEqual(
                uow.artifacts.find_by_checksum(self.artifact.checksum_sha256),
                self.artifact,
            )
            self.assertEqual(
                uow.processing_versions.get(self.processing.processing_version_id),
                self.processing,
            )
            self.assertEqual(
                uow.regions.get(self.region.region_version_id), self.region
            )
            self.assertEqual(
                uow.tec_observations.for_artifact(self.artifact.artifact_id),
                (self.tec,),
            )
            self.assertEqual(
                uow.tec_observations.for_artifact_at(
                    self.artifact.artifact_id,
                    self.tec.observed_at,
                ),
                (self.tec,),
            )
            self.assertEqual(
                uow.tec_observations.for_artifact_at(
                    self.artifact.artifact_id,
                    self.tec.observed_at + timedelta(seconds=1),
                ),
                (),
            )
            self.assertEqual(
                uow.regional_observations.for_artifact(self.artifact.artifact_id),
                (prior_observation, self.observation, self.actual),
            )
            self.assertEqual(
                uow.regional_observations.get(self.observation.observation_id),
                self.observation,
            )
            self.assertEqual(
                uow.regional_observations.find(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at=self.observation.observed_at,
                ),
                self.observation,
            )
            self.assertEqual(uow.models.get(self.model.model_version_id), self.model)
            self.assertEqual(
                uow.detectors.get(self.detector.detector_version_id), self.detector
            )
            stored_forecast = replace(forecast, status=ForecastStatus.SCORED)
            self.assertEqual(
                uow.forecasts.get(self.forecast.forecast_id), stored_forecast
            )
            self.assertEqual(
                uow.forecasts.find_equivalent(forecast), stored_forecast
            )
            self.assertEqual(
                uow.decisions.for_forecast(self.forecast.forecast_id),
                (self.decision, confirming_decision),
            )
            self.assertEqual(
                uow.decisions.search(
                    valid_at_or_after=forecast.valid_at - timedelta(minutes=1),
                    valid_before=forecast.valid_at + timedelta(minutes=1),
                    region_version_id=self.region.region_version_id,
                    is_forecast_anomaly=True,
                    disturbance_assessment=DisturbanceAssessment.CONFIRMED,
                ),
                (confirming_decision,),
            )

    def test_identical_writes_are_idempotent_but_conflicts_are_rejected(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_through_forecast(uow)

            self._add_dependencies(uow)
            uow.tec_observations.add_many((self.tec, self.tec))
            uow.regional_observations.add(self.observation)
            uow.forecasts.add(self.forecast)

            with self.assertRaises(sqlite3.IntegrityError):
                uow.models.register(replace(self.model, model_type="not-persistence"))
            with self.assertRaisesRegex(ValueError, "must be set together"):
                uow.models.register(
                    replace(self.model, artifact_checksum_sha256=None)
                )
            with self.assertRaises(sqlite3.IntegrityError):
                uow.artifacts.add(
                    replace(self.artifact, artifact_id="same-bytes-new-id")
                )

            equivalent = replace(
                self.forecast,
                forecast_id="retry-generated-another-id",
                predicted_vtec_tecu=999.0,
            )
            self.assertEqual(uow.forecasts.find_equivalent(equivalent), self.forecast)
            with self.assertRaises(sqlite3.IntegrityError):
                uow.forecasts.add(equivalent)
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.tec_observations.for_artifact(self.artifact.artifact_id),
                (self.tec,),
            )

    def test_same_content_can_have_distinct_acquisition_provenance(self) -> None:
        alternate = replace(
            self.artifact,
            artifact_id="artifact-same-content-other-source",
            provider="other-provider",
            source_uri="https://other.example.test/the-same-file.ionex",
        )
        with SQLiteUnitOfWork(self.database) as uow:
            uow.artifacts.add(self.artifact)
            uow.artifacts.add(alternate)
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(uow.artifacts.get(alternate.artifact_id), alternate)
            self.assertEqual(
                uow.artifacts.find_acquisition(
                    provider=alternate.provider,
                    product=alternate.product,
                    parser_version=alternate.parser_version,
                    revision_priority=alternate.revision_priority,
                    revision=alternate.revision,
                    source_uri=alternate.source_uri,
                    checksum_sha256=alternate.checksum_sha256,
                ),
                alternate,
            )

    def test_forecasts_with_different_input_manifests_remain_distinct(self) -> None:
        alternate_observation = replace(
            self.observation,
            observation_id="observation-alternate-input",
            observed_at=T0 - timedelta(hours=1),
        )
        alternate_forecast = replace(
            self.forecast,
            forecast_id="forecast-alternate-input",
            predicted_vtec_tecu=11.0,
            input_observation_ids=(alternate_observation.observation_id,),
        )
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_through_forecast(uow)
            uow.regional_observations.add(alternate_observation)
            uow.forecasts.add(alternate_forecast)
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.forecasts.find_equivalent(self.forecast),
                self.forecast,
            )
            self.assertEqual(
                uow.forecasts.find_equivalent(alternate_forecast),
                alternate_forecast,
            )

    def test_regional_history_is_inclusive_limited_and_revision_aware(self) -> None:
        older = replace(
            self.observation,
            observation_id="observation-earlier",
            observed_at=T0 - timedelta(hours=1),
            mean_vtec_tecu=11.0,
            median_vtec_tecu=10.9,
        )
        revised_artifact = replace(
            self.artifact,
            artifact_id="artifact-revised",
            revision_priority=20,
            checksum_sha256="b" * 64,
            storage_ref="artifacts/artifact-revised.ionex",
            ingested_at=self.artifact.ingested_at + timedelta(days=1),
            revision="r2",
        )
        revised = replace(
            self.observation,
            observation_id="observation-revised",
            artifact_id=revised_artifact.artifact_id,
            produced_at=revised_artifact.ingested_at,
            mean_vtec_tecu=13.2,
            median_vtec_tecu=13.0,
        )
        alternate_processing = replace(
            self.processing,
            processing_version_id="processing-experimental",
            configuration_hash="experimental-processing-config-hash",
        )
        cross_processing_candidate = replace(
            revised,
            observation_id="observation-other-processing",
            processing_version_id=alternate_processing.processing_version_id,
            mean_vtec_tecu=99.0,
            median_vtec_tecu=99.0,
        )
        late_backfill_artifact = replace(
            self.artifact,
            artifact_id="artifact-late-backfill",
            revision_priority=5,
            checksum_sha256="e" * 64,
            storage_ref="artifacts/artifact-late-backfill.ionex",
            ingested_at=self.artifact.ingested_at + timedelta(days=2),
            revision="late-backfill",
        )
        late_backfill = replace(
            self.observation,
            observation_id="observation-late-backfill",
            artifact_id=late_backfill_artifact.artifact_id,
            produced_at=late_backfill_artifact.ingested_at,
            mean_vtec_tecu=1.0,
            median_vtec_tecu=1.0,
        )
        other_series_artifact = replace(
            self.artifact,
            artifact_id="artifact-other-series",
            provider="other-provider",
            product="other-product",
            revision_priority=100,
            checksum_sha256="f" * 64,
            storage_ref="artifacts/artifact-other-series.ionex",
            ingested_at=self.artifact.ingested_at + timedelta(days=3),
            revision="other-series",
        )
        other_series = replace(
            self.observation,
            observation_id="observation-other-series",
            artifact_id=other_series_artifact.artifact_id,
            produced_at=other_series_artifact.ingested_at,
            mean_vtec_tecu=100.0,
            median_vtec_tecu=100.0,
        )

        with SQLiteUnitOfWork(self.database) as uow:
            self._add_dependencies(uow)
            uow.processing_versions.register(alternate_processing)
            uow.artifacts.add(revised_artifact)
            uow.artifacts.add(late_backfill_artifact)
            uow.artifacts.add(other_series_artifact)
            uow.regional_observations.add(older)
            uow.regional_observations.add(self.observation)
            uow.regional_observations.add(revised)
            uow.regional_observations.add(cross_processing_candidate)
            uow.regional_observations.add(late_backfill)
            uow.regional_observations.add(other_series)
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.regional_observations.find(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at=T0,
                ),
                revised,
            )
            self.assertEqual(
                uow.regional_observations.history(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at_or_before=T0,
                ),
                (older, revised),
            )
            self.assertEqual(
                uow.regional_observations.history(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at_or_before=T0,
                    available_at_or_before=(
                        self.artifact.ingested_at + timedelta(hours=1)
                    ),
                ),
                (older, self.observation),
            )
            self.assertEqual(
                uow.regional_observations.history(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at_or_before=T0,
                    limit=1,
                ),
                (revised,),
            )
            self.assertEqual(
                uow.regional_observations.for_artifact(self.artifact.artifact_id),
                (older, self.observation),
            )
            self.assertEqual(
                uow.regional_observations.history(
                    source_provider=self.artifact.provider,
                    source_product=self.artifact.product,
                    region_version_id=self.region.region_version_id,
                    processing_version_id=self.processing.processing_version_id,
                    observed_at_or_before=T0,
                    limit=0,
                ),
                (),
            )

    def test_delayed_processing_does_not_change_source_revision_precedence(
        self,
    ) -> None:
        delayed_artifact = replace(
            self.artifact,
            artifact_id="artifact-older-delayed-processing",
            checksum_sha256="1" * 64,
            storage_ref="artifacts/older-delayed.ionex",
            ingested_at=T0 + timedelta(minutes=30),
            revision="correction-a",
        )
        newer_artifact = replace(
            self.artifact,
            artifact_id="artifact-newer-source",
            checksum_sha256="2" * 64,
            storage_ref="artifacts/newer-source.ionex",
            ingested_at=T0 + timedelta(hours=1),
            revision="correction-b",
        )
        delayed = replace(
            self.observation,
            observation_id="observation-older-delayed-processing",
            artifact_id=delayed_artifact.artifact_id,
            produced_at=T0 + timedelta(hours=2),
            median_vtec_tecu=1.0,
        )
        newer = replace(
            self.observation,
            observation_id="observation-newer-source",
            artifact_id=newer_artifact.artifact_id,
            produced_at=T0 + timedelta(hours=1, minutes=1),
            median_vtec_tecu=2.0,
        )
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_dependencies(uow)
            uow.artifacts.add(delayed_artifact)
            uow.artifacts.add(newer_artifact)
            uow.regional_observations.add(delayed)
            uow.regional_observations.add(newer)
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            selected = uow.regional_observations.find(
                source_provider=self.artifact.provider,
                source_product=self.artifact.product,
                region_version_id=self.region.region_version_id,
                processing_version_id=self.processing.processing_version_id,
                observed_at=T0,
                available_at_or_before=T0 + timedelta(hours=3),
            )
        self.assertEqual(selected, newer)

    def test_durable_boundaries_reject_semantically_impossible_links(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_dependencies(uow)
            with self.assertRaisesRegex(ValueError, "VTEC must be finite"):
                uow.tec_observations.add_many(
                    (replace(self.tec, vtec_tecu=float("nan")),)
                )
            uow.regional_observations.add(self.observation)
            uow.regional_observations.add(self.actual)

            with self.assertRaisesRegex(ValueError, "before it is observed"):
                uow.regional_observations.add(
                    replace(
                        self.observation,
                        observation_id="observation-produced-too-early",
                        produced_at=T0 - timedelta(hours=1),
                    )
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "source series"):
                uow.forecasts.add(
                    replace(
                        self.forecast,
                        forecast_id="forecast-wrong-series",
                        source_provider="wrong-provider",
                    )
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "after forecast origin"):
                uow.forecasts.add(
                    replace(
                        self.forecast,
                        forecast_id="forecast-future-input",
                        input_observation_ids=(self.actual.observation_id,),
                    )
                )

            uow.forecasts.add(self.forecast)
            wrong_time_decision = replace(
                self.decision,
                decision_id="decision-wrong-target-time",
                observation_id=self.observation.observation_id,
                residual_tecu=0.0,
                anomaly_score=0.0,
                is_forecast_anomaly=False,
                disturbance_assessment=DisturbanceAssessment.NORMAL,
            )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "valid_at"):
                uow.decisions.add(wrong_time_decision)
            with self.assertRaisesRegex(sqlite3.IntegrityError, "threshold"):
                uow.decisions.add(
                    replace(
                        self.decision,
                        decision_id="decision-wrong-threshold",
                        threshold=self.decision.threshold + 0.25,
                    )
                )

            uow.decisions.add(self.decision)
            uow.commit()

    def test_forecast_lifecycle_replacement_only_changes_status(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            self._add_through_forecast(uow)
            self.assertEqual(
                uow.forecasts.list_pending(
                    valid_at_or_before=self.forecast.valid_at
                ),
                (self.forecast,),
            )
            scored = replace(self.forecast, status=ForecastStatus.SCORED)
            uow.forecasts.replace(scored)
            uow.forecasts.replace(scored)

            with self.assertRaises(ValueError):
                uow.forecasts.replace(self.forecast)
            with self.assertRaises(sqlite3.IntegrityError):
                uow.forecasts.replace(
                    replace(scored, predicted_vtec_tecu=100.0)
                )
            self.assertEqual(
                uow.forecasts.list_pending(
                    valid_at_or_before=self.forecast.valid_at + timedelta(days=1)
                ),
                (),
            )
            uow.commit()

    def test_uncommitted_and_exceptional_units_roll_back(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            uow.artifacts.add(self.artifact)

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertIsNone(uow.artifacts.get(self.artifact.artifact_id))

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with SQLiteUnitOfWork(self.database) as uow:
                uow.artifacts.add(self.artifact)
                raise RuntimeError("stop")

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertIsNone(uow.artifacts.get(self.artifact.artifact_id))
            uow.artifacts.add(self.artifact)
            uow.rollback()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertIsNone(uow.artifacts.get(self.artifact.artifact_id))

    def test_add_many_rolls_its_savepoint_back_if_caller_catches_conflict(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            uow.artifacts.add(self.artifact)
            uow.tec_observations.add_many((self.tec,))
            uow.commit()

        another = replace(
            self.tec,
            latitude_degrees=31.0,
            quality_flags=("new",),
        )
        conflicting = replace(self.tec, vtec_tecu=self.tec.vtec_tecu + 1.0)
        with SQLiteUnitOfWork(self.database) as uow:
            with self.assertRaises(sqlite3.IntegrityError):
                uow.tec_observations.add_many((another, conflicting))
            uow.commit()

        with SQLiteUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.tec_observations.for_artifact(self.artifact.artifact_id),
                (self.tec,),
            )

    def test_schema_initialization_is_repeatable_and_foreign_keys_are_enforced(self) -> None:
        with SQLiteUnitOfWork(self.database) as uow:
            uow.commit()
        with SQLiteUnitOfWork(self.database) as uow:
            version = uow._require_connection().execute(  # test schema metadata
                "PRAGMA user_version"
            ).fetchone()[0]
            self.assertEqual(version, SCHEMA_VERSION)
            with self.assertRaises(sqlite3.IntegrityError):
                uow.tec_observations.add_many((self.tec,))

    def test_unknown_schema_version_is_rejected(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA user_version = 999")
        connection.close()

        with self.assertRaisesRegex(RuntimeError, "unsupported database schema"):
            with SQLiteUnitOfWork(self.database):
                pass

    def test_rejects_naive_datetimes(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            replace(self.artifact, ingested_at=T0.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
