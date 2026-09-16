"""Administrative commands for the PostgreSQL and Zarr data plane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from ophanim.postgres_catalog import PostgresTECCatalog
from ophanim.tec_archive import ZarrTECGridStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize and inspect OPHANIM's bulk TEC data plane",
    )
    parser.add_argument(
        "--dsn",
        default="",
        help=(
            "PostgreSQL connection string; empty uses standard PGHOST, PGPORT, "
            "PGDATABASE, PGUSER, and PGPASSWORD environment variables"
        ),
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get("OPHANIM_ZARR_ROOT", "var/zarr")
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init",
        help="create or validate the PostgreSQL schema and Zarr root",
    )
    subparsers.add_parser(
        "status",
        help="print catalog counts and the configured Zarr root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    catalog: PostgresTECCatalog | None = None
    try:
        grid_store = ZarrTECGridStore(arguments.zarr_root)
        with tempfile.NamedTemporaryFile(
            dir=grid_store.root,
            prefix=".ophanim-write-probe-",
        ) as probe:
            probe.write(b"ophanim-zarr-write-probe\n")
            probe.flush()
            os.fsync(probe.fileno())
        catalog = PostgresTECCatalog(arguments.dsn)
        catalog.wait()
        catalog.initialize_schema()
        if arguments.command == "init":
            payload = {
                "ok": True,
                "postgres_schema": "ophanim",
                "zarr_format": 3,
                "zarr_root": str(grid_store.root),
            }
        else:
            payload = {
                "ok": True,
                "postgres_schema": "ophanim",
                "zarr_root": str(grid_store.root),
                "catalog": catalog.statistics().to_dict(),
            }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ophanim-data: {error}", file=sys.stderr)
        return 1
    finally:
        if catalog is not None:
            catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
