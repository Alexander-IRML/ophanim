"""Focused tests for aggregation, persistence forecasting, and detection."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from ophanim.aggregation import (
    AggregationError,
    AggregationPolicy,
    BoundaryDefinitionError,
    BoundsRegionSelector,
    GeographicBounds,
    InsufficientRegionalDataError,
    MeanMedianRegionalAggregator,
    PolygonBoundary,
    PolygonRegionSelector,
    boundary_checksum_sha256,
)
from ophanim.detection import (
    CalibrationConfigurationError,
    ForecastObservationMismatchError,
    InsufficientObservationDataError,
    ResidualCalibration,
    ResidualZScoreDecisionEngine,
)
from ophanim.domain import (
    DetectorVersion,
    DisturbanceAssessment,
    Forecast,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    TECObservation,
    TargetMetric,
)
from ophanim.forecasting import (
    AmbiguousHistoryError,
    ForecastRequest,
    HistoryLeakageError,
    PersistenceForecaster,
    PersistenceModelRunner,
)


UTC = timezone.utc
EPOCH = datetime(2026, 1, 2, 12, tzinfo=UTC)


def processing_version() -> ProcessingVersion:
    return ProcessingVersion(
        processing_version_id="processing-v1",
        code_revision="abc123",
        configuration_hash="processing-config",
        parser_version="test-parser/1",
        created_at=EPOCH - timedelta(days=2),
    )


def region_version(
    boundary_ref: str = "central-texas-v1",
    boundary=None,
) -> RegionVersion:
    boundary = boundary or GeographicBounds(30.0, -100.0, 32.0, -98.0)
    return RegionVersion(
        region_version_id="region-v1",
        name="Central Texas",
        boundary_ref=boundary_ref,
        boundary_checksum_sha256=boundary_checksum_sha256(boundary),
        created_at=EPOCH - timedelta(days=2),
    )


def grid_observation(
    latitude: float,
    longitude: float,
    value: float,
    *,
    observed_at: datetime = EPOCH,
    flags: tuple[str, ...] = (),
) -> TECObservation:
    return TECObservation(
        artifact_id="artifact-v1",
        observed_at=observed_at,
        latitude_degrees=latitude,
        longitude_degrees=longitude,
        vtec_tecu=value,
        quality_flags=flags,
    )


def regional_observation(
    observation_id: str,
    observed_at: datetime,
    *,
    mean: float = 12.0,
    median: float = 11.0,
    coverage: float = 1.0,
    flags: tuple[str, ...] = (),
    produced_at: datetime | None = None,
) -> RegionalObservation:
    return RegionalObservation(
        observation_id=observation_id,
        artifact_id=f"artifact-{observation_id}",
        processing_version_id="processing-v1",
        region_version_id="region-v1",
        observed_at=observed_at,
        produced_at=produced_at or observed_at,
        mean_vtec_tecu=mean,
        median_vtec_tecu=median,
        cell_count=4,
        coverage_fraction=coverage,
        quality_flags=flags,
    )


def persistence_model() -> ModelVersion:
    return ModelVersion(
        model_version_id="persistence-v1",
        model_type="persistence",
        configuration_hash="model-config",
        created_at=EPOCH - timedelta(days=1),
    )


def pending_forecast() -> Forecast:
    return Forecast(
        forecast_id="forecast-v1",
        origin_at=EPOCH,
        issued_at=EPOCH,
        valid_at=EPOCH + timedelta(hours=2),
        source_provider="test-provider",
        source_product="test-product",
        region_version_id="region-v1",
        model_version_id="persistence-v1",
        processing_version_id="processing-v1",
        target_metric=TargetMetric.MEDIAN_VTEC,
        predicted_vtec_tecu=10.0,
        status=ForecastStatus.PENDING,
        input_observation_ids=("history-v1",),
    )


def detector_version(**overrides: object) -> DetectorVersion:
    values: dict[str, object] = {
        "detector_version_id": "residual-v1",
        "scoring_method": "residual_zscore",
        "threshold": 2.0,
        "configuration_hash": "detector-config",
        "created_at": EPOCH,
        "calibration_data_version": "calibration-v1",
        "calibration_residual_mean_tecu": 0.0,
        "calibration_residual_standard_deviation_tecu": 2.0,
        "calibration_sample_count": 100,
    }
    values.update(overrides)
    return DetectorVersion(**values)


class RegionalAggregationTests(unittest.TestCase):
    def test_bounds_strategy_computes_mean_median_and_ignores_outside_cells(
        self,
    ) -> None:
        selector = BoundsRegionSelector(
            {
                "central-texas-v1": GeographicBounds(
                    south_latitude_degrees=30.0,
                    west_longitude_degrees=-100.0,
                    north_latitude_degrees=32.0,
                    east_longitude_degrees=-98.0,
                )
            }
        )
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=selector,
            policy=AggregationPolicy(expected_cell_count=4),
        )
        inputs = (
            grid_observation(30.0, -100.0, 10.0),
            grid_observation(30.0, -99.0, 12.0),
            grid_observation(31.0, -100.0, 100.0),
            grid_observation(32.0, -98.0, 14.0),
            grid_observation(40.0, -99.0, 999.0),
        )

        result = aggregator.aggregate(
            observations=inputs,
            region=region_version(),
            processing_version=processing_version(),
            produced_at=EPOCH,
        )

        self.assertEqual(result.mean_vtec_tecu, 34.0)
        self.assertEqual(result.median_vtec_tecu, 13.0)
        self.assertEqual(result.cell_count, 4)
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertEqual(result.quality_flags, ())
        repeated = aggregator.aggregate(
            observations=inputs,
            region=region_version(),
            processing_version=processing_version(),
            produced_at=EPOCH,
        )
        self.assertEqual(repeated.observation_id, result.observation_id)

    def test_quality_exclusion_and_coverage_are_explicit(self) -> None:
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=BoundsRegionSelector(
                {
                    "central-texas-v1": GeographicBounds(
                        30.0, -100.0, 32.0, -98.0
                    )
                }
            ),
            policy=AggregationPolicy(
                expected_cell_count=4,
                minimum_coverage_fraction=0.5,
                preferred_coverage_fraction=0.75,
            ),
        )

        result = aggregator.aggregate(
            observations=(
                grid_observation(30.0, -100.0, 10.0),
                grid_observation(31.0, -99.0, 14.0),
                grid_observation(32.0, -98.0, math.nan, flags=("missing",)),
            ),
            region=region_version(),
            processing_version=processing_version(),
            produced_at=EPOCH,
        )

        self.assertEqual(result.mean_vtec_tecu, 12.0)
        self.assertEqual(result.median_vtec_tecu, 12.0)
        self.assertEqual(result.coverage_fraction, 0.5)
        self.assertEqual(result.cell_count, 2)
        self.assertEqual(
            result.quality_flags,
            (
                "excluded_cell:missing",
                "excluded_cell:non_finite_vtec",
                "excluded_cells",
                "low_coverage",
                "missing_cells",
            ),
        )

    def test_below_minimum_coverage_is_insufficient_data(self) -> None:
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=BoundsRegionSelector(
                {"central-texas-v1": GeographicBounds(30, -100, 32, -98)}
            ),
            policy=AggregationPolicy(
                expected_cell_count=4, minimum_coverage_fraction=0.75
            ),
        )
        with self.assertRaises(InsufficientRegionalDataError):
            aggregator.aggregate(
                observations=(grid_observation(31, -99, 10),),
                region=region_version(),
                processing_version=processing_version(),
                produced_at=EPOCH,
            )

    def test_region_version_rejects_a_changed_boundary_definition(self) -> None:
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=BoundsRegionSelector(
                {"central-texas-v1": GeographicBounds(30, -100, 32, -98)}
            ),
            policy=AggregationPolicy(expected_cell_count=1),
        )
        changed_region = RegionVersion(
            region_version_id="region-v1",
            name="Central Texas",
            boundary_ref="central-texas-v1",
            boundary_checksum_sha256="0" * 64,
            created_at=EPOCH - timedelta(days=2),
        )

        with self.assertRaisesRegex(BoundaryDefinitionError, "checksum"):
            aggregator.aggregate(
                observations=(grid_observation(31, -99, 10),),
                region=changed_region,
                processing_version=processing_version(),
                produced_at=EPOCH,
            )

    def test_polygon_includes_edges_and_excludes_outside_points(self) -> None:
        boundary = PolygonBoundary(((0.0, 0.0), (0.0, 2.0), (2.0, 0.0)))
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=PolygonRegionSelector({"triangle-v1": boundary}),
            policy=AggregationPolicy(expected_cell_count=2),
        )
        result = aggregator.aggregate(
            observations=(
                grid_observation(0.0, 0.0, 2.0),
                grid_observation(0.5, 0.5, 4.0),
                grid_observation(1.5, 1.5, 100.0),
            ),
            region=region_version("triangle-v1", boundary),
            processing_version=processing_version(),
            produced_at=EPOCH,
        )
        self.assertEqual(result.cell_count, 2)
        self.assertEqual(result.mean_vtec_tecu, 3.0)

    def test_mixed_timestamps_and_duplicate_cells_are_rejected(self) -> None:
        aggregator = MeanMedianRegionalAggregator(
            selection_strategy=BoundsRegionSelector(
                {"central-texas-v1": GeographicBounds(30, -100, 32, -98)}
            ),
            policy=AggregationPolicy(expected_cell_count=2),
        )
        with self.assertRaises(AggregationError):
            aggregator.aggregate(
                observations=(
                    grid_observation(31, -99, 10),
                    grid_observation(
                        32, -98, 11, observed_at=EPOCH + timedelta(minutes=15)
                    ),
                ),
                region=region_version(),
                processing_version=processing_version(),
                produced_at=EPOCH,
            )
        with self.assertRaises(AggregationError):
            aggregator.aggregate(
                observations=(
                    grid_observation(31, -99, 10),
                    grid_observation(31, -99, 11),
                ),
                region=region_version(),
                processing_version=processing_version(),
                produced_at=EPOCH,
            )

    def test_wrapping_bounds_handle_the_antimeridian(self) -> None:
        bounds = GeographicBounds(-10, 170, 10, -170)
        self.assertTrue(bounds.contains(0, 175))
        self.assertTrue(bounds.contains(0, -175))
        self.assertTrue(bounds.contains(0, -180))
        self.assertFalse(bounds.contains(0, 0))

        dateline_edge = GeographicBounds(-10, 170, 10, -180)
        self.assertTrue(dateline_edge.contains(0, 175))
        self.assertTrue(dateline_edge.contains(0, -180))
        self.assertFalse(dateline_edge.contains(0, -179))

    def test_boundaries_enforce_half_open_longitude_convention(self) -> None:
        with self.assertRaisesRegex(BoundaryDefinitionError, r"\[-180, 180\)"):
            GeographicBounds(-10, 170, 10, 180)
        with self.assertRaisesRegex(BoundaryDefinitionError, r"\[-180, 180\)"):
            PolygonBoundary(((0, 180), (1, 179), (-1, 179)))


class PersistenceForecastTests(unittest.TestCase):
    def test_runner_uses_latest_observation_and_issues_pending_forecast(self) -> None:
        older = regional_observation("old", EPOCH - timedelta(hours=1), median=8)
        latest = regional_observation("latest", EPOCH, median=11)
        request = ForecastRequest(
            origin_at=EPOCH,
            issued_at=EPOCH,
            valid_at=EPOCH + timedelta(hours=2),
            source_provider="test-provider",
            source_product="test-product",
            region_version_id="region-v1",
            target_metric=TargetMetric.MEDIAN_VTEC,
        )
        runner = PersistenceModelRunner()

        result = runner.run(
            request=request,
            history=(older, latest),
            model_version=persistence_model(),
            processing_version=processing_version(),
        )

        self.assertEqual(result.predicted_vtec_tecu, 11.0)
        self.assertEqual(result.input_observation_ids, ("latest",))
        self.assertIs(result.status, ForecastStatus.PENDING)
        repeated = runner.run(
            request=request,
            history=(older, latest),
            model_version=persistence_model(),
            processing_version=processing_version(),
        )
        self.assertEqual(repeated.forecast_id, result.forecast_id)

    def test_future_history_is_rejected_as_leakage(self) -> None:
        request = ForecastRequest(
            origin_at=EPOCH,
            issued_at=EPOCH,
            valid_at=EPOCH + timedelta(hours=2),
            source_provider="test-provider",
            source_product="test-product",
            region_version_id="region-v1",
            target_metric=TargetMetric.MEAN_VTEC,
        )
        with self.assertRaises(HistoryLeakageError):
            PersistenceModelRunner().run(
                request=request,
                history=(regional_observation("future", request.valid_at),),
                model_version=persistence_model(),
                processing_version=processing_version(),
            )

    def test_same_instant_revisions_require_an_explicit_choice(self) -> None:
        history = (
            regional_observation("revision-a", EPOCH),
            regional_observation("revision-b", EPOCH),
        )
        with self.assertRaises(AmbiguousHistoryError):
            PersistenceForecaster().predict(
                history=history,
                model_version=persistence_model(),
                target_metric=TargetMetric.MEDIAN_VTEC,
            )


class ResidualDetectionTests(unittest.TestCase):
    def actual_observation(
        self,
        *,
        median: float,
        flags: tuple[str, ...] = (),
        observed_at: datetime | None = None,
    ) -> RegionalObservation:
        return regional_observation(
            "actual-v1",
            observed_at or pending_forecast().valid_at,
            median=median,
            coverage=0.75 if flags else 1.0,
            flags=flags,
        )

    def test_threshold_crossing_is_candidate_not_confirmation(self) -> None:
        decision = ResidualZScoreDecisionEngine().assess(
            forecast=pending_forecast(),
            observation=self.actual_observation(median=16.0),
            detector_version=detector_version(),
            scored_at=EPOCH + timedelta(hours=2, minutes=1),
        )
        self.assertEqual(decision.residual_tecu, 6.0)
        self.assertEqual(decision.anomaly_score, 3.0)
        self.assertTrue(decision.is_forecast_anomaly)
        self.assertIs(
            decision.disturbance_assessment, DisturbanceAssessment.CANDIDATE
        )

    def test_non_anomaly_is_normal(self) -> None:
        decision = ResidualZScoreDecisionEngine().assess(
            forecast=pending_forecast(),
            observation=self.actual_observation(median=12.0),
            detector_version=detector_version(),
            scored_at=EPOCH + timedelta(hours=3),
        )
        self.assertEqual(decision.anomaly_score, 1.0)
        self.assertFalse(decision.is_forecast_anomaly)
        self.assertIs(decision.disturbance_assessment, DisturbanceAssessment.NORMAL)

    def test_uncertain_quality_keeps_score_but_assessment_unknown(self) -> None:
        decision = ResidualZScoreDecisionEngine().assess(
            forecast=pending_forecast(),
            observation=self.actual_observation(
                median=16.0, flags=("low_coverage",)
            ),
            detector_version=detector_version(),
            scored_at=EPOCH + timedelta(hours=3),
        )
        self.assertTrue(decision.is_forecast_anomaly)
        self.assertIs(
            decision.disturbance_assessment, DisturbanceAssessment.UNKNOWN
        )

    def test_disqualifying_quality_does_not_produce_a_decision(self) -> None:
        with self.assertRaises(InsufficientObservationDataError):
            ResidualZScoreDecisionEngine().assess(
                forecast=pending_forecast(),
                observation=self.actual_observation(
                    median=16.0, flags=("invalid",)
                ),
                detector_version=detector_version(),
                scored_at=EPOCH + timedelta(hours=3),
            )

    def test_observation_must_match_forecast_target_instant(self) -> None:
        with self.assertRaises(ForecastObservationMismatchError):
            ResidualZScoreDecisionEngine().assess(
                forecast=pending_forecast(),
                observation=self.actual_observation(
                    median=16.0,
                    observed_at=pending_forecast().valid_at + timedelta(minutes=15),
                ),
                detector_version=detector_version(),
                scored_at=EPOCH + timedelta(hours=3),
            )

    def test_detector_requires_durable_calibration_parameters(self) -> None:
        with self.assertRaises(CalibrationConfigurationError):
            ResidualZScoreDecisionEngine().assess(
                forecast=pending_forecast(),
                observation=self.actual_observation(median=16.0),
                detector_version=detector_version(
                    calibration_residual_standard_deviation_tecu=None
                ),
                scored_at=EPOCH + timedelta(hours=3),
            )

    def test_calibration_helper_uses_sample_standard_deviation(self) -> None:
        calibration = ResidualCalibration.from_residuals((-1.0, 1.0))
        self.assertEqual(calibration.residual_mean_tecu, 0.0)
        self.assertAlmostEqual(
            calibration.residual_standard_deviation_tecu, math.sqrt(2.0)
        )
        self.assertEqual(calibration.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
