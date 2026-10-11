"""Abstract base class, state machine, error hierarchy, data contracts, and audio utilities."""
import os
import re
import json
import logging
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, TypedDict, Generator

import numpy as np

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
StopCheck = Callable[[], bool]


# ── Error hierarchy ──
class BackendError(RuntimeError): pass
class BackendLoadError(BackendError): pass
class BackendGenerateError(BackendError): pass
class BackendCancelled(BackendError): pass


# ── State machine ──
class BackendState(Enum):
    UNLOADED = auto()
    LOADING = auto()
    READY = auto()
    GENERATING = auto()
    ERROR = auto()


# ── Capabilities ──
@dataclass
class BackendCapabilities:
    supports_seed: bool = False
    supports_batch: bool = True
    supports_streaming: bool = False
    min_vram_gb: float = 0.0
    max_duration_sec: int = 300
    output_sample_rates: List[int] = field(default_factory=lambda: [44100])
    description: str = ""


# ── Data contracts ──
class SettingItem(TypedDict, total=False):
    id: str
    type: str
    label: str
    description: str
    category: str
    default: Any
    min: float
    max: float
    step: float
    options: List[str]
    placeholder: str


class PromptSchema(TypedDict, total=False):
    required: bool
    title: Optional[str]
    description: Optional[str]
    placeholder: Optional[str]


class BackendMetadata(TypedDict, total=False):
    name: str
    description: str
    details: str
    icon: str
    requirements: str


class GenerationResult(TypedDict):
    audio: Any
    sample_rate: int


# ════════════════════════════════════════════════════════════════
#  AudioUtils
# ════════════════════════════════════════════════════════════════
class AudioUtils:
    """Generic audio manipulation utilities."""

    @staticmethod
    def to_numpy(audio: Any) -> np.ndarray:
        """Convert torch tensors or lists to a float32 numpy array."""
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 0:
            return audio.reshape(1)
        if audio.ndim > 1:
            return np.mean(audio, axis=-1)
        return audio

    @staticmethod
    def to_stereo(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return np.stack([audio, audio], axis=0)
        return audio

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if not _HAS_LIBROSA:
            logger.warning("librosa not installed — skipping resample")
            return audio
        if audio.ndim > 1:
            return librosa.resample(audio.T, orig_sr=orig_sr, target_sr=target_sr).T
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def safe_normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """Normalize with zero-division protection."""
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-8:
            return audio
        return (audio * (target_peak / peak)).astype(np.float32)


# ── Abstract backend ──
class MusicBackend(ABC):
    def __init__(self):
        self._state = BackendState.UNLOADED

    @property
    def state(self) -> BackendState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == BackendState.READY

    def _set_state(self, state: BackendState) -> None:
        old = self._state
        self._state = state
        if old != state:
            logger.debug("%s: %s → %s", type(self).__name__, old.name, state.name)

    @abstractmethod
    def _load_model(self, config) -> None: ...

    @abstractmethod
    def _generate(
        self, prompt: str, duration: int, params: dict,
        progress_callback: Optional[ProgressCallback],
        stop_check: Optional[StopCheck],
    ) -> Optional[GenerationResult]: ...

    @abstractmethod
    def _unload(self) -> None: ...

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities()

    def load_model(self, config) -> None:
        self._set_state(BackendState.LOADING)
        try:
            self._validate_config(config)
            self._load_model(config)
            self._set_state(BackendState.READY)
        except BackendError:
            self._set_state(BackendState.ERROR)
            raise
        except Exception as exc:
            self._set_state(BackendState.ERROR)
            raise BackendLoadError(str(exc)) from exc

    def generate(
        self, prompt: str, duration: int, params: dict,
        progress_callback: Optional[ProgressCallback] = None,
        stop_check: Optional[StopCheck] = None,
    ) -> Optional[GenerationResult]:
        if not self.is_ready:
            raise BackendGenerateError(f"Backend not ready (state={self._state.name})")
        self._set_state(BackendState.GENERATING)
        try:
            result = self._generate(prompt, duration, params, progress_callback, stop_check)
            self._set_state(BackendState.READY)
            return result
        except BackendCancelled:
            self._set_state(BackendState.READY)
            return None
        except BackendError:
            self._set_state(BackendState.ERROR)
            raise
        except Exception as exc:
            self._set_state(BackendState.ERROR)
            raise BackendGenerateError(str(exc)) from exc

    def stream_generate(self, prompt, duration, params, chunk_size=1024):
        if not type(self).get_capabilities().supports_streaming:
            raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        yield from self._stream_generate(prompt, duration, params, chunk_size)

    def _stream_generate(self, prompt, duration, params, chunk_size):
        raise NotImplementedError

    def unload(self) -> None:
        try:
            self._unload()
        finally:
            self._set_state(BackendState.UNLOADED)

    def terminate(self) -> None:
        """Called by the UI to forcefully cancel a blocking generation."""
        pass

    @classmethod
    def validate_config(cls, config) -> List[str]:
        return []

    def _validate_config(self, config) -> None:
        errors = type(self).validate_config(config)
        if errors:
            raise BackendLoadError("Config validation failed: " + "; ".join(errors))

    @classmethod
    def get_settings_schema(cls) -> List[SettingItem]:
        return []

    @classmethod
    def get_prompt_schema(cls) -> PromptSchema:
        return {
            "required": True,
            "title": "📝 Prompt",
            "description": "Enter a text prompt for generation.",
            "placeholder": "Enter your prompt here...",
        }

    @classmethod
    def get_examples(cls) -> List[dict]:
        """Return a list of example prompts and parameters."""
        return []

    @classmethod
    def get_metadata(cls) -> BackendMetadata:
        return {
            "name": cls.__name__,
            "description": "",
            "details": "",
            "icon": "📦",
            "requirements": "",
        }

    @property
    def display_name(self) -> str:
        return str(self.get_metadata().get("name") or type(self).__name__)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.get_metadata.__func__ is MusicBackend.get_metadata.__func__:
            logger.warning("Backend %s does not override get_metadata()", cls.__name__)


# ════════════════════════════════════════════════════════════════
#  AudioProcessor — normalisation + equal-power fade
# ════════════════════════════════════════════════════════════════
class AudioProcessor:
    """Post-processing: peak normalisation and equal-power fade in/out."""

    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            target = 10.0 ** (target_db / 20.0)
            return (audio * (target / peak)).astype(np.float32)
        return audio

    @staticmethod
    def apply_fade(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
        """Apply equal-power fade in and out."""
        if fade_ms <= 0 or audio.size == 0:
            return audio
        fade_samples = min(int(sr * fade_ms / 1000.0), audio.shape[0] // 2)
        if fade_samples <= 0:
            return audio

        t = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
        fade_in = np.sin(t)
        fade_out = np.cos(t)

        if audio.ndim == 1:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        else:
            audio[:fade_samples] *= fade_in[:, np.newaxis]
            audio[-fade_samples:] *= fade_out[:, np.newaxis]
        return audio

    @staticmethod
    def process(result: GenerationResult, config) -> GenerationResult:
        audio = AudioUtils.to_numpy(result["audio"])
        sr = int(result["sample_rate"])
        if audio.size == 0:
            return {"audio": audio, "sample_rate": sr}

        pp = {}
        if hasattr(config, "post_processing"):
            pp = config.post_processing or {}

        if pp.get("normalize", True):
            audio = AudioProcessor.normalize(audio)

        fade_ms = pp.get("fade_ms", 0)
        if fade_ms and fade_ms > 0:
            audio = AudioProcessor.apply_fade(audio, sr, fade_ms)

        return {"audio": audio, "sample_rate": sr}


# ════════════════════════════════════════════════════════════════
#  AudioAnalyzer
# ════════════════════════════════════════════════════════════════
class AudioAnalyzer:
    """Compute descriptive statistics for a GenerationResult."""

    NEG_INF = -float("inf")

    @staticmethod
    def _db(x: float) -> float:
        return 20 * np.log10(x) if x > 1e-10 else AudioAnalyzer.NEG_INF

    @staticmethod
    def analyze(result: GenerationResult) -> dict:
        audio = AudioUtils.to_numpy(result["audio"])
        sr = int(result["sample_rate"])
        if audio.size == 0:
            return {
                "duration_sec": 0,
                "peak_db": AudioAnalyzer.NEG_INF,
                "rms_db": AudioAnalyzer.NEG_INF,
                "dynamic_range_db": 0,
                "crest_factor_db": 0,
                "sample_rate": sr,
                "n_samples": 0,
            }

        mono = AudioUtils.to_mono(audio)
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono ** 2)))

        peak_db = AudioAnalyzer._db(peak)
        rms_db = AudioAnalyzer._db(rms)
        crest_db = (peak_db - rms_db) if (peak > 1e-10 and rms > 1e-10) else 0.0

        return {
            "duration_sec": round(len(mono) / sr, 2),
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "dynamic_range_db": round(crest_db, 2),
            "crest_factor_db": round(crest_db, 2),
            "sample_rate": sr,
            "n_samples": int(len(mono)),
        }


# ════════════════════════════════════════════════════════════════
#  AudioExporter
# ════════════════════════════════════════════════════════════════
class AudioExporter:
    """Export GenerationResult to WAV/FLAC/OGG with JSON metadata sidecar."""

    SUPPORTED_FORMATS = ("WAV", "FLAC", "OGG")
    _FORMAT_SUBTYPES = {
        "WAV": "PCM_16",
        "FLAC": "PCM_16",
        "OGG": "VORBIS",
    }

    @staticmethod
    def _slugify(text: str, max_len: int = 50) -> str:
        slug = re.sub(r"[^\w\s-]", "", text.lower())[:max_len].strip()
        slug = re.sub(r"[\s_-]+", "_", slug)
        return slug or "audio"

    def make_filename(self, prompt: str, variation: int, seed: int) -> str:
        slug = self._slugify(prompt)
        parts = [slug, f"v{variation:02d}"]
        if seed:
            parts.append(f"s{seed}")
        return "_".join(parts)

    @staticmethod
    def _ensure_unique(path: str) -> str:
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(f"{stem}_{i}{ext}"):
            i += 1
        return f"{stem}_{i}{ext}"

    def export(
        self,
        result: GenerationResult,
        output_dir: str,
        filename: str,
        metadata: dict,
        fmt: str = "WAV",
        backend_name: str = "",
        organize: bool = True,
    ) -> str:
        if not _HAS_SOUNDFILE:
            raise BackendError("soundfile is not installed — run: pip install soundfile")

        fmt = (fmt or "WAV").upper()
        if fmt not in self.SUPPORTED_FORMATS:
            logger.warning("Unsupported format '%s', falling back to WAV", fmt)
            fmt = "WAV"

        base_dir = output_dir
        if organize and backend_name:
            base_dir = os.path.join(output_dir, backend_name)
        date_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)

        audio = AudioUtils.to_numpy(result["audio"])
        sr = int(result["sample_rate"])

        if audio.ndim == 1:
            pass
        elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T

        ext = "wav" if fmt == "WAV" else fmt.lower()
        subtype = self._FORMAT_SUBTYPES[fmt]

        out_path = os.path.join(date_dir, f"{filename}.{ext}")
        out_path = self._ensure_unique(out_path)

        try:
            sf.write(out_path, audio, sr, subtype=subtype)
        except Exception as exc:
            raise BackendError(f"Failed to write audio file: {exc}") from exc

        meta_path = os.path.splitext(out_path)[0] + ".json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to write metadata sidecar: %s", exc)

        logger.info("Exported audio → %s", out_path)
        return out_path