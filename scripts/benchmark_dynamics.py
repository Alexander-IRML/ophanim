"""Emit a reproducible confidence/error sensitivity report, not calibration.

Run from the repository: python scripts/benchmark_dynamics.py --output report.json
The report is a generated experiment artifact, never a sensing observation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    # Keep this small numerical experiment laptop-friendly unless the caller
    # explicitly chose another thread budget before launching it.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.test_science_acceptance import confidence_sweep, wave_sensitivity_sweep
    from tests.test_dynamics_reliability import reliability_sensitivity_sweep
    payload = confidence_sweep()
    payload["wave_sensitivity"] = wave_sensitivity_sweep()
    payload["reliability_sensitivity"] = reliability_sensitivity_sweep()
    repository = Path(__file__).resolve().parents[1]
    sources = sorted((repository / "src/ophanim/dynamics").glob("*.py"))
    sources += [repository / "tests/test_science_acceptance.py",
                repository / "tests/test_dynamics_reliability.py", Path(__file__).resolve()]
    hashes = {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    payload["provenance"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_sha256": hashes,
        "software": {name: version(name) for name in ("numpy", "scipy", "xarray", "scikit-image", "pyproj")},
        "python": sys.version,
        "semantics": "Hashes include the working source files, not only a possibly stale Git commit",
    }
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(content, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(content, encoding="utf-8")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
