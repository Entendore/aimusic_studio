"""Music-generation backend package."""
import logging
from typing import List

from .base import (
    MusicBackend, SettingItem, PromptSchema, BackendMetadata,
    GenerationResult, ProgressCallback, StopCheck,
    BackendError, BackendLoadError, BackendGenerateError, BackendCancelled,
    BackendState, BackendCapabilities,
    AudioProcessor, AudioExporter, AudioAnalyzer, AudioUtils,
)

logger = logging.getLogger(__name__)


def _load_backends():
    """Lazily import backends, tolerating missing dependencies."""
    registry = {}
    try:
        from .math_synth import MathSynthBackend
        registry["mathsynth"] = MathSynthBackend
    except Exception as exc:
        logger.warning("Failed to import MathSynthBackend: %s", exc)

    try:
        from .heartlib import HeartMuLaBackend
        registry["heartmula"] = HeartMuLaBackend
    except Exception as exc:
        logger.warning("Failed to import HeartMuLaBackend: %s", exc)

    try:
        from .acestep import ACEStepBackend
        registry["acestep"] = ACEStepBackend
    except Exception as exc:
        logger.warning("Failed to import ACEStepBackend: %s", exc)

    return registry


BACKEND_REGISTRY = _load_backends()

__all__ = [
    "MusicBackend", "SettingItem", "PromptSchema", "BackendMetadata",
    "GenerationResult", "ProgressCallback", "StopCheck",
    "BackendError", "BackendLoadError", "BackendGenerateError", "BackendCancelled",
    "BackendState", "BackendCapabilities",
    "AudioProcessor", "AudioExporter", "AudioAnalyzer", "AudioUtils",
    "BACKEND_REGISTRY",
]


def get_backend(name: str) -> MusicBackend:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name]()


def list_backends() -> List[str]:
    return list(BACKEND_REGISTRY.keys())