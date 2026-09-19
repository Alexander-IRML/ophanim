"""Compatibility facade for the former science/art application composition.

New code imports scientific publication from ophanim.core.runs and artistic
publication from ophanim.shawtynet.runs. Optional art dependencies are lazy.
"""
from __future__ import annotations

from ophanim.core.artifacts import (RUN_SCHEMA, canonical_json, dataset_sha256,
    file_sha256, json_value, verify_run, write_json, _publish, _write_dataset)
from ophanim.core.runs import analyze_run


def software_identity(*, artistic=False):
    from ophanim.core.artifacts import software_identity as identity
    return identity(extra_sections=("shawtynet",) if artistic else ())


def visualize_run(*args, **kwargs):
    from ophanim.shawtynet.runs import visualize_run as operation
    return operation(*args, **kwargs)


def render_run(*args, **kwargs):
    from ophanim.shawtynet.runs import render_run as operation
    return operation(*args, **kwargs)


def composite_run(*args, **kwargs):
    from ophanim.shawtynet.runs import composite_run as operation
    return operation(*args, **kwargs)
