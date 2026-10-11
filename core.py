from __future__ import annotations

import os
import logging
import threading
import datetime
from typing import Optional, List, Dict, Any, Callable

from backend import (
    MusicBackend, BackendError, BackendCancelled,
    BackendMetadata, BackendCapabilities,
    AudioProcessor, AudioExporter, AudioAnalyzer,
    GenerationResult, BACKEND_REGISTRY,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

class HardwareManager:
    def __init__(self, config):
        self.requested_device = config.device
        self.active_device = self._validate_device()

    def _validate_device(self) -> str:
        if not _HAS_TORCH:
            return "cpu"
        requested = self.requested_device
        if requested == "auto":
            if torch.cuda.is_available(): return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
            return "cpu"
        if requested == "cuda" and torch.cuda.is_available(): return "cuda"
        if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def get_context(self):
        return torch.device(self.active_device) if _HAS_TORCH else None

    def clear_cache(self):
        if not _HAS_TORCH: return
        if self.active_device == "cuda": torch.cuda.empty_cache()
        elif self.active_device == "mps": torch.mps.empty_cache()

class PluginRegistry:
    _cache_lock = threading.Lock()
    _class_cache: Optional[Dict[str, type]] = None
    _metadata_cache: Optional[Dict[str, BackendMetadata]] = None
    _capabilities_cache: Optional[Dict[str, BackendCapabilities]] = None

    @classmethod
    def _ensure_cache(cls):
        with cls._cache_lock:
            if cls._class_cache is not None: return
            cls._class_cache, cls._metadata_cache, cls._capabilities_cache = {}, {}, {}
            for name, plugin_cls in BACKEND_REGISTRY.items():
                try:
                    cls._class_cache[name] = plugin_cls
                    cls._metadata_cache[name] = plugin_cls.get_metadata()
                    cls._capabilities_cache[name] = plugin_cls.get_capabilities()
                except Exception as exc:
                    logger.error("Failed to load backend '%s': %s", name, exc)

    @classmethod
    def list_available(cls) -> List[str]:
        cls._ensure_cache()
        return sorted(cls._class_cache.keys())

    @classmethod
    def get_all_metadata(cls) -> Dict[str, BackendMetadata]:
        cls._ensure_cache()
        return dict(cls._metadata_cache)

    @classmethod
    def get_all_capabilities(cls) -> Dict[str, BackendCapabilities]:
        cls._ensure_cache()
        return dict(cls._capabilities_cache)

    @classmethod
    def get_class(cls, name: str):
        cls._ensure_cache()
        if name not in cls._class_cache:
            raise BackendError(f"Unknown backend: {name}")
        return cls._class_cache[name]

class GenerationService:
    PHASES = {
        "generate":        (0, 85),
        "post-processing": (85, 92),
        "analyzing":       (92, 97),
        "saving":          (97, 100),
    }

    def __init__(self, backend_service: "BackendService", exporter: AudioExporter):
        self._backend_svc = backend_service
        self._exporter = exporter

    def _make_phase_callback(self, phase_key: str, outer_cb):
        lo, hi = self.PHASES[phase_key]
        def _cb(p: int, phase_label: str = ""):
            if outer_cb is None: return
            clamped = max(0, min(100, p))
            overall = int(lo + (hi - lo) * (clamped / 100.0))
            outer_cb(overall, phase_label or phase_key.title())
        return _cb

    def run_single(self, prompt, duration, params, output_dir, variation,
                   export_format="WAV", organize_output=True,
                   progress_callback=None, stop_check=None) -> Optional[dict]:
        backend = self._backend_svc.active
        if not backend or not backend.is_ready:
            raise BackendError("No ready backend")

        backend_name = self._backend_svc.active_name
        config = self._backend_svc.config

        gen_cb = self._make_phase_callback("generate", progress_callback)
        try:
            result = backend.generate(prompt, duration, params, progress_callback=gen_cb, stop_check=stop_check)
        except BackendCancelled:
            return None

        if result is None: return None
        if stop_check and stop_check(): return None

        if progress_callback: progress_callback(self.PHASES["post-processing"][0], "Post-processing")
        result = AudioProcessor.process(result, config)

        if progress_callback: progress_callback(self.PHASES["analyzing"][0], "Analyzing audio")
        analysis = AudioAnalyzer.analyze(result)

        if stop_check and stop_check(): return None

        if progress_callback: progress_callback(self.PHASES["saving"][0], "Saving to disk")
        seed = int(params.get("seed", 0))
        filename = self._exporter.make_filename(prompt, variation, seed)
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "backend": backend_name, "prompt": prompt, "duration": duration,
            "params": params, "analysis": analysis,
            "device": config.device,
        }
        path = self._exporter.export(result, output_dir, filename, metadata,
                                     fmt=export_format, backend_name=backend_name, organize=organize_output)
        return {"path": path, "analysis": analysis}

class BackendService:
    def __init__(self, config):
        self._config = config
        self._active: Optional[MusicBackend] = None
        self._active_name: str = ""

    @property
    def active(self) -> Optional[MusicBackend]: return self._active
    @property
    def active_name(self) -> str: return self._active_name
    @property
    def config(self): return self._config
    @property
    def capabilities(self) -> BackendCapabilities:
        return type(self._active).get_capabilities() if self._active else BackendCapabilities()

    def load(self, name: str):
        if name not in PluginRegistry.list_available():
            raise BackendError(f"Backend '{name}' not found")

        if self._active is not None:
            try: self._active.unload()
            except Exception as exc: logger.warning("Error during unload: %s", exc)
            self._active = None
            self._active_name = ""

        try:
            cls = PluginRegistry.get_class(name)
            instance = cls()
            instance.load_model(self.config)
            self._active = instance
            self._active_name = name
        except BackendError:
            raise
        except ImportError as exc:
            raise BackendError(f"Missing dependencies for {name}: {exc}") from exc
        except Exception as exc:
            raise BackendError(f"Failed to initialize {name}: {exc}") from exc

    def cancel_generation(self):
        """Forcefully cancel the active backend's generation."""
        if self._active and hasattr(self._active, 'terminate'):
            self._active.terminate()

    def unload(self):
        if self._active:
            self._active.unload()
            self._active = None
            self._active_name = ""

class AppContext:
    def __init__(self, config):
        self.config = config
        self.hardware = HardwareManager(config)
        self.config.set_system_path("device", self.hardware.active_device)
        self.backend_service = BackendService(config)
        self._exporter = AudioExporter()
        self.generation_service = GenerationService(self.backend_service, self._exporter)
        self.available_backends = PluginRegistry.list_available()
        self.backend_metadata = PluginRegistry.get_all_metadata()
        self.backend_capabilities = PluginRegistry.get_all_capabilities()

    def shutdown(self):
        try:
            if self.backend_service.active:
                self.backend_service.unload()
        except Exception as exc:
            logger.warning("Error during shutdown: %s", exc)