"""Smoke test for the package skeleton."""

import unittest


class PackageImportTests(unittest.TestCase):
    def test_public_modules_import(self) -> None:
        from ophanim import (  # noqa: F401
            aggregation,
            artifacts,
            bootstrap,
            data_cli,
            detection,
            forecasting,
            gim,
            ingestion,
            ionex,
            mamba_model,
            mamba_monitor,
            postgres_catalog,
            reconciliation,
            regional_grid,
            repositories,
            sqlite,
            tec_archive,
            workflows,
        )
        from ophanim import domain  # noqa: F401


if __name__ == "__main__":
    unittest.main()
