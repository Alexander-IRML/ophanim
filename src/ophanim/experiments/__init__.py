"""Independent experiments: scenario fixtures, model benchmarks, future agent seams.

Experiments consume the reusable core and never depend on artistic rendering.
Agent contracts deliberately do not imply an installed or trained agent system.
"""

from .contracts import (DatasetSplit, ExperimentSpec, MetricResult, AgentEnvironment,
                        AgentTrainer, agent_training_capability)
from .scenarios import ScenarioConfig, scenario_from_candidate, make_scenario
from .hypotheses import HypothesisConfig, hypothesis_from_evidence, alternatives_from_evidence


def simulate_hypothesis(*args, **kwargs):
    from .volume_model import simulate_hypothesis as simulate
    return simulate(*args, **kwargs)


def publish_hypothesis_run(*args, **kwargs):
    from .volume_model import publish_hypothesis_run as publish
    return publish(*args, **kwargs)


def make_synthetic_dataset(*args, **kwargs):
    from .synthetic import make_synthetic_dataset as make

    return make(*args, **kwargs)


def degrade_dataset(*args, **kwargs):
    from .synthetic import degrade_dataset as degrade

    return degrade(*args, **kwargs)


__all__ = ["make_synthetic_dataset", "degrade_dataset", "ScenarioConfig", "scenario_from_candidate",
           "make_scenario", "DatasetSplit", "ExperimentSpec", "MetricResult", "AgentEnvironment",
           "AgentTrainer", "agent_training_capability", "HypothesisConfig", "hypothesis_from_evidence",
           "alternatives_from_evidence", "simulate_hypothesis", "publish_hypothesis_run"]
