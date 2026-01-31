# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

# Suppress noisy output from dependencies at import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import litellm

litellm.suppress_debug_info = True

from .adapters import default_adapter
from .api import optimize
from .cache import configure_cache, disable_cache
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.callbacks import LiteLLMCacheLogger, VerboseCallback
from .core.result import GEPAResult
from .examples import aime
from .logging.logger import get_logger
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

__all__ = [
    "aime",
    "CompositeStopper",
    "configure_cache",
    "default_adapter",
    "disable_cache",
    "EvaluationBatch",
    "FileStopper",
    "GEPAAdapter",
    "GEPAResult",
    "get_logger",
    "LiteLLMCacheLogger",
    "MaxMetricCallsStopper",
    "NoImprovementStopper",
    "optimize",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    "TimeoutStopCondition",
    "VerboseCallback",
]
