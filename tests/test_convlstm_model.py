from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta

from ophanim.convlstm_model import (
    ALGORITHM,
    ARTIFACT_FORMAT,
    TORCH_AVAILABLE,
    CircularLongitudeConv2d,
    ConvLSTMArtifactError,
    ConvLSTMConfig,
    ConvLSTMError,
    ConvLSTMUnavailableError,
    GridNormalization,
    build_causal_batch,
    create_model,
    create_optimizer,
    fit_grid_normalization,
    load_checkpoint,
    load_model,
    masked_l1_loss,
    predict_next,
    serialize_checkpoint,
    serialize_model,
    train_batch,
)

if TORCH_AVAILABLE:  # Keep the core test suite importable without the extra.
    import torch
else:  # pragma: no cover - only used to make annotations harmless.
    torch = None


_DIGEST = "a" * 64
_START = datetime(2024, 1, 1, tzinfo=UTC)


def _config() -> ConvLSTMConfig:
    return ConvLSTMConfig(
        grid_height=4,
        grid_width=5,
        hidden_channels=2,
        kernel_size=3,
        initialization_seed=37,
    )


class ConvLSTMContractTests(unittest.TestCase):
    def test_default_contract_is_native_grid_and_prior_day_residual(self) -> None:
        config = ConvLSTMConfig()

        self.assertEqual((config.grid_height, config.grid_width), (71, 72))
        self.assertEqual(config.history_frames, 12)
        self.assertEqual(config.reference_lag_frames, 12)
        self.assertEqual(
            config.reference_lag_frames * config.cadence_seconds,
            86_400,
        )
        self.assertEqual(config.reference_history_index, 0)
        self.assertEqual(config, ConvLSTMConfig.from_bytes(config.to_bytes()))
        self.assertEqual(len(config.configuration_sha256), 64)

    def test_configuration_bytes_are_canonical_and_strict(self) -> None:
        config = ConvLSTMConfig()
        pretty = json.dumps(config.to_dict(), indent=2).encode("utf-8")
        with self.assertRaisesRegex(ConvLSTMArtifactError, "canonical"):
            ConvLSTMConfig.from_bytes(pretty)

        extended = config.to_dict()
        extended["future_option"] = True
        encoded = json.dumps(
            extended,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with self.assertRaisesRegex(ConvLSTMArtifactError, "fields"):
            ConvLSTMConfig.from_bytes(encoded)

    def test_configuration_rejects_non_daily_reference(self) -> None:
        with self.assertRaisesRegex(ConvLSTMError, "exactly one UTC day"):
            ConvLSTMConfig(reference_lag_frames=11)

    @unittest.skipIf(TORCH_AVAILABLE, "unavailable-runtime behavior only")
    def test_optional_runtime_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(
            ConvLSTMUnavailableError,
            r"ophanim\[spatial\]",
        ):
            create_model()


@unittest.skipUnless(TORCH_AVAILABLE, "requires optional PyTorch dependency")
class ConvLSTMTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _config()
        self.normalization = GridNormalization(center_tecu=20.0, scale_tecu=5.0)
        base = torch.arange(14 * 4 * 5, dtype=torch.float32).reshape(14, 4, 5)
        self.vtec = 8.0 + base / 25.0
        self.valid = torch.ones_like(self.vtec, dtype=torch.bool)
        self.times = tuple(
            _START + timedelta(seconds=self.config.cadence_seconds * index)
            for index in range(len(self.vtec))
        )

    def test_causal_batch_uses_prior_day_reference_and_never_target_as_input(
        self,
    ) -> None:
        ordinary = build_causal_batch(
            self.vtec,
            self.valid,
            (12,),
            config=self.config,
            normalization=self.normalization,
            observed_at=self.times,
        )
        changed_values = self.vtec.clone()
        changed_values[12] += 500.0
        changed = build_causal_batch(
            changed_values,
            self.valid,
            (12,),
            config=self.config,
            normalization=self.normalization,
            observed_at=self.times,
        )

        self.assertEqual(tuple(ordinary.inputs.shape), (1, 12, 2, 4, 5))
        self.assertTrue(torch.equal(ordinary.inputs, changed.inputs))
        self.assertTrue(torch.equal(ordinary.reference_vtec, self.vtec[0][None, None]))
        self.assertTrue(torch.equal(ordinary.reference_vtec, changed.reference_vtec))
        self.assertFalse(torch.equal(ordinary.target_residual, changed.target_residual))

    def test_missing_values_are_masked_from_inputs_and_loss(self) -> None:
        values = self.vtec.clone()
        valid = self.valid.clone()
        values[5, 1, 2] = float("nan")
        valid[5, 1, 2] = False
        values[12, 0, 0] = float("nan")
        valid[12, 0, 0] = False
        batch = build_causal_batch(
            values,
            valid,
            (12,),
            config=self.config,
            normalization=self.normalization,
        )

        self.assertEqual(float(batch.inputs[0, 5, 0, 1, 2]), 0.0)
        self.assertEqual(float(batch.inputs[0, 5, 1, 1, 2]), 0.0)
        self.assertFalse(bool(batch.loss_mask[0, 0, 0, 0]))
        prediction = torch.zeros_like(batch.target_residual)
        self.assertTrue(torch.isfinite(masked_l1_loss(
            prediction,
            batch.target_residual,
            batch.loss_mask,
        )))

    def test_circular_longitude_does_not_wrap_latitude(self) -> None:
        values = torch.tensor(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
            dtype=torch.float32,
        )
        longitude = CircularLongitudeConv2d(1, 1, 3, bias=False)
        longitude.convolution.weight.data.zero_()
        longitude.convolution.weight.data[0, 0, 1, 0] = 1.0
        longitude_result = longitude(values)
        self.assertEqual(float(longitude_result[0, 0, 0, 0].detach()), 3.0)
        self.assertEqual(float(longitude_result[0, 0, 1, 0].detach()), 6.0)

        latitude = CircularLongitudeConv2d(1, 1, 3, bias=False)
        latitude.convolution.weight.data.zero_()
        latitude.convolution.weight.data[0, 0, 0, 1] = 1.0
        latitude_result = latitude(values)
        self.assertEqual(float(latitude_result[0, 0, 0, 1].detach()), 0.0)
        self.assertEqual(float(latitude_result[0, 0, 1, 1].detach()), 2.0)

    def test_prediction_is_deterministic_and_has_native_map_shape(self) -> None:
        first = create_model(self.config)
        second = create_model(self.config)
        for first_value, second_value in zip(
            first.state_dict().values(),
            second.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(first_value, second_value))

        prediction = predict_next(
            first,
            self.vtec[:12],
            self.valid[:12],
            config=self.config,
            normalization=self.normalization,
        )
        self.assertEqual(tuple(prediction.predicted_vtec_tecu.shape), (1, 1, 4, 5))
        self.assertEqual(tuple(prediction.predicted_residual_tecu.shape), (1, 1, 4, 5))
        self.assertTrue(torch.equal(prediction.valid_mask, self.valid[0][None, None]))

    def test_masked_l1_ignores_invalid_cells_and_rejects_empty_mask(self) -> None:
        prediction = torch.tensor([[[[1.0, 1000.0]]]])
        target = torch.tensor([[[[3.0, -1000.0]]]])
        mask = torch.tensor([[[[True, False]]]])

        self.assertEqual(float(masked_l1_loss(prediction, target, mask)), 2.0)
        with self.assertRaisesRegex(ConvLSTMError, "at least one valid"):
            masked_l1_loss(prediction, target, torch.zeros_like(mask))

    def test_robust_normalization_uses_only_valid_cells(self) -> None:
        values = torch.tensor([[1.0, 2.0, 1000.0], [3.0, 4.0, 5.0]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        normalization = fit_grid_normalization(values, valid)

        self.assertEqual(normalization.center_tecu, 3.0)
        self.assertAlmostEqual(normalization.scale_tecu, 1.4826, places=5)

    def test_canonical_artifact_round_trip_preserves_prediction(self) -> None:
        model = create_model(self.config)
        before = predict_next(
            model,
            self.vtec[:12],
            self.valid[:12],
            config=self.config,
            normalization=self.normalization,
        )
        artifact = serialize_model(
            model,
            config=self.config,
            normalization=self.normalization,
            training_data_sha256=_DIGEST,
            completed_epochs=3,
        )
        restored = load_model(artifact)
        after = predict_next(
            restored.model,
            self.vtec[:12],
            self.valid[:12],
            config=restored.config,
            normalization=restored.normalization,
        )

        self.assertEqual(artifact, serialize_model(
            model,
            config=self.config,
            normalization=self.normalization,
            training_data_sha256=_DIGEST,
            completed_epochs=3,
        ))
        self.assertEqual(restored.model.algorithm, ALGORITHM)
        self.assertEqual(json.loads(artifact)["artifact_format"], ARTIFACT_FORMAT)
        self.assertTrue(torch.equal(
            before.predicted_vtec_tecu,
            after.predicted_vtec_tecu,
        ))

    def test_checkpoint_restores_optimizer_for_exact_next_step(self) -> None:
        model = create_model(self.config)
        optimizer = create_optimizer(model, self.config)
        first_batch = build_causal_batch(
            self.vtec,
            self.valid,
            (12,),
            config=self.config,
            normalization=self.normalization,
        )
        second_batch = build_causal_batch(
            self.vtec,
            self.valid,
            (13,),
            config=self.config,
            normalization=self.normalization,
        )
        train_batch(model, optimizer, first_batch)
        checkpoint = serialize_checkpoint(
            model,
            optimizer,
            config=self.config,
            normalization=self.normalization,
            training_data_sha256=_DIGEST,
            completed_epochs=2,
            next_window_index=9,
        )
        restored = load_checkpoint(checkpoint)

        self.assertEqual(restored.completed_epochs, 2)
        self.assertEqual(restored.next_window_index, 9)
        original_loss = train_batch(model, optimizer, second_batch)
        restored_loss = train_batch(
            restored.model,
            restored.optimizer,
            second_batch,
        )
        self.assertEqual(original_loss, restored_loss)
        for original, recovered in zip(
            model.state_dict().values(),
            restored.model.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(original, recovered))


if __name__ == "__main__":
    unittest.main()
