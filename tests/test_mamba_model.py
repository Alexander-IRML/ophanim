from __future__ import annotations

import json
import math
import unittest
from datetime import UTC, datetime, timedelta

from ophanim.mamba_model import (
    ALGORITHM,
    MambaArtifactError,
    MambaModel,
    MambaModelError,
    TrainingSample,
    fit_model,
)


_START = datetime(2006, 1, 1, tzinfo=UTC)
_STEP = timedelta(hours=2)


def _sample(index: int, *, offset: int = 0, spike: float = 0.0) -> TrainingSample:
    phase = math.tau * (index + offset) / 12.0
    seasonal = math.tau * (index + offset) / (12.0 * 91.0)
    mean = 18.0 + 5.0 * math.sin(phase) + 1.5 * math.cos(seasonal)
    median = 16.5 + 4.25 * math.sin(phase - 0.08) + math.cos(seasonal)
    return TrainingSample(
        observed_at=_START + (index + offset) * _STEP,
        mean_vtec_tecu=mean + spike,
        median_vtec_tecu=median + spike * 0.9,
        coverage_fraction=0.96,
    )


def _training(count: int = 180) -> tuple[TrainingSample, ...]:
    return tuple(_sample(index) for index in range(count))


def _gap_fill(
    sample: TrainingSample,
    *,
    mean_vtec_tecu: float | None = None,
    median_vtec_tecu: float | None = None,
) -> TrainingSample:
    return TrainingSample(
        observed_at=sample.observed_at,
        mean_vtec_tecu=(
            sample.mean_vtec_tecu
            if mean_vtec_tecu is None
            else mean_vtec_tecu
        ),
        median_vtec_tecu=(
            sample.median_vtec_tecu
            if median_vtec_tecu is None
            else median_vtec_tecu
        ),
        coverage_fraction=1e-6,
    )


class MambaReferenceModelTests(unittest.TestCase):
    def test_fit_is_deterministic_and_identifies_honest_algorithm(self) -> None:
        first = fit_model(_training())
        second = fit_model(_training())

        self.assertEqual(first.algorithm, "ophanim-mamba-selective-ssm/1")
        self.assertEqual(first.algorithm, ALGORITHM)
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)
        self.assertEqual(len(first.artifact_sha256), 64)

    def test_canonical_artifact_round_trip_preserves_scores(self) -> None:
        model = fit_model(_training())
        blob = model.to_bytes()
        restored = MambaModel.from_bytes(blob)
        targets = tuple(_sample(index) for index in range(180, 184))

        self.assertEqual(restored, model)
        self.assertEqual(restored.to_bytes(), blob)
        self.assertEqual(
            restored.score_sequence((), targets),
            model.score_sequence((), targets),
        )
        payload = json.loads(blob)
        self.assertEqual(payload["algorithm"], ALGORITHM)
        self.assertFalse(blob.endswith(b"\n"))

    def test_noncanonical_or_nonfinite_artifacts_are_rejected(self) -> None:
        model = fit_model(_training())
        payload = json.loads(model.to_bytes())
        pretty = json.dumps(payload, indent=2).encode("utf-8")
        with self.assertRaisesRegex(MambaArtifactError, "not canonical"):
            MambaModel.from_bytes(pretty)

        payload["configuration"]["threshold"] = float("nan")
        malformed = json.dumps(
            payload,
            allow_nan=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with self.assertRaises(MambaModelError):
            MambaModel.from_bytes(malformed)

    def test_large_unseen_spike_is_an_anomaly_candidate(self) -> None:
        model = fit_model(_training(), threshold=4.5)
        normal = _sample(180)
        spike = TrainingSample(
            observed_at=normal.observed_at,
            mean_vtec_tecu=normal.mean_vtec_tecu + 60.0,
            median_vtec_tecu=normal.median_vtec_tecu + 55.0,
            coverage_fraction=normal.coverage_fraction,
        )

        result = model.score_sequence((), (spike,))[0]

        self.assertTrue(result.is_candidate)
        self.assertGreaterEqual(result.anomaly_score, result.threshold)
        self.assertTrue(result.mean_is_candidate or result.median_is_candidate)
        self.assertEqual(result.to_dict()["is_candidate"], True)

    def test_target_value_is_not_used_until_after_its_prediction(self) -> None:
        model = fit_model(_training())
        normal_first = _sample(180)
        changed_first = TrainingSample(
            observed_at=normal_first.observed_at,
            mean_vtec_tecu=normal_first.mean_vtec_tecu + 75.0,
            median_vtec_tecu=normal_first.median_vtec_tecu + 70.0,
            coverage_fraction=normal_first.coverage_fraction,
        )
        normal_prediction = model.score_sequence((), (normal_first,))[0]
        changed_prediction = model.score_sequence((), (changed_first,))[0]

        self.assertEqual(
            normal_prediction.predicted_mean_vtec_tecu,
            changed_prediction.predicted_mean_vtec_tecu,
        )
        self.assertEqual(
            normal_prediction.predicted_median_vtec_tecu,
            changed_prediction.predicted_median_vtec_tecu,
        )
        self.assertNotEqual(
            normal_prediction.mean_anomaly_score,
            changed_prediction.mean_anomaly_score,
        )

    def test_prior_target_changes_only_later_prediction(self) -> None:
        model = fit_model(_training())
        first = _sample(180)
        second = _sample(181)
        changed_first = TrainingSample(
            observed_at=first.observed_at,
            mean_vtec_tecu=first.mean_vtec_tecu + 40.0,
            median_vtec_tecu=first.median_vtec_tecu + 35.0,
            coverage_fraction=first.coverage_fraction,
        )

        normal = model.score_sequence((), (first, second))
        changed = model.score_sequence((), (changed_first, second))

        self.assertEqual(
            normal[0].predicted_mean_vtec_tecu,
            changed[0].predicted_mean_vtec_tecu,
        )
        self.assertNotEqual(
            normal[1].predicted_mean_vtec_tecu,
            changed[1].predicted_mean_vtec_tecu,
        )

    def test_history_replays_already_seen_post_training_readouts(self) -> None:
        model = fit_model(_training())
        first = _sample(180)
        second = _sample(181)
        together = model.score_sequence((), (first, second))[1]
        replayed = model.score_sequence((first,), (second,))[0]
        self.assertEqual(together, replayed)

    def test_gap_fill_advances_state_but_is_not_fit_or_calibration_target(
        self,
    ) -> None:
        ordinary_gaps = list(_training())
        extreme_gaps = list(_training())
        for index in range(160, 180):
            ordinary_gaps[index] = _gap_fill(ordinary_gaps[index])
            extreme_gaps[index] = _gap_fill(
                extreme_gaps[index],
                mean_vtec_tecu=5_000.0 + index,
                median_vtec_tecu=4_000.0 + index,
            )

        ordinary = fit_model(ordinary_gaps)
        extreme = fit_model(extreme_gaps)

        # There are 159 genuinely observed next-readout targets; the final 20
        # gap fills are not counted in the chronological calibration split.
        self.assertEqual(ordinary.calibration_sample_count, 32)
        self.assertEqual(extreme.calibration_sample_count, 32)
        self.assertEqual(ordinary.mean_readout, extreme.mean_readout)
        self.assertEqual(ordinary.median_readout, extreme.median_readout)
        self.assertEqual(
            ordinary.mean_residual_center_tecu,
            extreme.mean_residual_center_tecu,
        )
        self.assertEqual(
            ordinary.mean_residual_scale_tecu,
            extreme.mean_residual_scale_tecu,
        )
        self.assertEqual(
            ordinary.median_residual_center_tecu,
            extreme.median_residual_center_tecu,
        )
        self.assertEqual(
            ordinary.median_residual_scale_tecu,
            extreme.median_residual_scale_tecu,
        )

        # The differing gap-fill values are nevertheless consumed by every
        # recurrent step and therefore produce a different final anchor state.
        self.assertNotEqual(ordinary.anchor_state, extreme.anchor_state)

    def test_robust_input_scaling_ignores_gap_fill_values(self) -> None:
        ordinary_gaps = list(_training())
        extreme_gaps = list(_training())
        for index in (30, 60, 90):
            ordinary_gaps[index] = _gap_fill(ordinary_gaps[index])
            extreme_gaps[index] = _gap_fill(
                extreme_gaps[index],
                mean_vtec_tecu=1_000_000.0 * index,
                median_vtec_tecu=750_000.0 * index,
            )

        ordinary = fit_model(ordinary_gaps)
        extreme = fit_model(extreme_gaps)

        self.assertEqual(ordinary.input_centers, extreme.input_centers)
        self.assertEqual(ordinary.input_scales, extreme.input_scales)
        self.assertNotEqual(ordinary.anchor_state, extreme.anchor_state)

    def test_too_few_observed_targets_reports_usable_target_shortfall(self) -> None:
        mostly_gaps = list(_training())
        for index in range(21, len(mostly_gaps)):
            mostly_gaps[index] = _gap_fill(mostly_gaps[index])

        with self.assertRaisesRegex(
            MambaModelError,
            r"20 usable supervised targets.*requires at least 18 fit targets",
        ):
            fit_model(mostly_gaps)

    def test_irregular_or_overlapping_sequences_fail_closed(self) -> None:
        model = fit_model(_training())
        with self.assertRaisesRegex(MambaModelError, "after the model"):
            model.score_sequence((_sample(179),), (_sample(180),))
        with self.assertRaisesRegex(MambaModelError, "consecutive"):
            model.score_sequence((), (_sample(181),))

        irregular = list(_training())
        irregular[50] = TrainingSample(
            observed_at=irregular[49].observed_at + timedelta(hours=3),
            mean_vtec_tecu=irregular[50].mean_vtec_tecu,
            median_vtec_tecu=irregular[50].median_vtec_tecu,
        )
        with self.assertRaises(MambaModelError):
            fit_model(irregular)


if __name__ == "__main__":
    unittest.main()
