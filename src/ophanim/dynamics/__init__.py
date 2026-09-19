"""Reusable TEC dynamics, independent of forecasting models and art backends.

The base OPHANIM installation remains dependency-free. Importing this package
loads only schemas; numerical dependencies are imported when analysis is used.
"""

from .schemas import AnalysisConfig, AnalysisError, AnalysisResult, FlowConfig, WaveConfig


def analyze_dataset(raw, config=None):
    """Analyze an observation snapshot without altering its input arrays."""
    from .pipeline import analyze_dataset as analyze

    return analyze(raw, config or AnalysisConfig())


def analyze_sequence(raw, config=None, frame_times=None, **options):
    """Analyze bounded exact-time windows without reusing a frozen event."""
    from .sequence import analyze_sequence as analyze
    return analyze(raw, config, frame_times, **options)


__all__ = ["AnalysisConfig", "AnalysisError", "AnalysisResult", "FlowConfig", "WaveConfig", "analyze_dataset", "analyze_sequence"]
