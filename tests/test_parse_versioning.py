"""Focused invariants tying raw parses to downstream processing."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

from ophanim.domain import ProcessingVersion, RegionVersion, SourceArtifact
from ophanim.sqlite import SQLiteUnitOfWork
from ophanim.workflows import RegionalAggregationWorkflow


UTC = timezone.utc
T0 = datetime(2026, 1, 2, tzinfo=UTC)


class _UnexpectedAggregator:
    def aggregate(self, **kwargs):
        raise AssertionError("mismatched raw parse reached the aggregator")


class RawParseVersioningTests(unittest.TestCase):
    def test_aggregation_rejects_raw_parse_not_declared_by_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "ophanim.sqlite3"
            artifact = SourceArtifact(
                artifact_id="artifact-parser-v1",
                provider="test-provider",
                product="test-product",
                parser_version="test-parser/1",
                revision_priority=0,
                checksum_sha256="a" * 64,
                storage_ref="artifacts/source.ionex",
                ingested_at=T0,
            )
            with SQLiteUnitOfWork(database) as unit_of_work:
                unit_of_work.artifacts.add(artifact)
                unit_of_work.commit()

            workflow = RegionalAggregationWorkflow(
                unit_of_work_factory=partial(SQLiteUnitOfWork, database),
                aggregator=_UnexpectedAggregator(),
                clock=lambda: T0 + timedelta(minutes=1),
            )
            processing = ProcessingVersion(
                processing_version_id="processing-parser-v2",
                code_revision="test-code",
                configuration_hash="test-config",
                parser_version="test-parser/2",
                created_at=T0,
            )
            region = RegionVersion(
                region_version_id="test-region-v1",
                name="Test region",
                boundary_ref="test-boundary-v1",
                boundary_checksum_sha256="b" * 64,
                created_at=T0,
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires parser 'test-parser/2'.*parsed with 'test-parser/1'",
            ):
                workflow.aggregate_artifact(
                    artifact_id=artifact.artifact_id,
                    region=region,
                    processing_version=processing,
                )


if __name__ == "__main__":
    unittest.main()
