# heartlib.py
import subprocess
import tempfile
import os
import sys
import time
import logging
from typing import Optional, List
import librosa
import numpy as np
from .base import (
    MusicBackend, SettingItem, PromptSchema, BackendMetadata,
    GenerationResult, ProgressCallback, StopCheck,
    BackendError, BackendLoadError, BackendGenerateError, BackendCancelled,
    BackendState, BackendCapabilities,
)

logger = logging.getLogger(__name__)


class HeartMuLaBackend(MusicBackend):
    """HeartMuLa music generation backend."""

    def __init__(self):
        super().__init__()
        self.heartlib_path = None
        self.model_path = None
        self.version = "3B"
        self._process = None

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            supports_seed=True,
            supports_batch=False,
            supports_streaming=False,
            min_vram_gb=8.0,
            max_duration_sec=240,
            output_sample_rates=[44100],
            description="HeartMuLa open-source music generation with lyrics and tags"
        )

    @classmethod
    def get_settings_schema(cls) -> List[SettingItem]:
        return [
            {"id": "heartlib_path", "type": "string", "label": "HeartLib Path",
             "description": "Path to the heartlib repository (contains examples/)",
             "default": "", "category": "Model"},
            {"id": "model_path", "type": "string", "label": "Model Checkpoint Path",
             "description": "Path to pretrained model checkpoints",
             "default": "", "category": "Model"},
            {"id": "version", "type": "select", "label": "Model Version",
             "description": "HeartMuLa model version", "default": "3B",
             "options": ["3B", "7B"], "category": "Model"},
            {"id": "topk", "type": "int", "label": "Top-K Sampling",
             "description": "Top-k sampling parameter", "default": 50,
             "min": 1, "max": 100, "category": "Generation"},
            {"id": "temperature", "type": "float", "label": "Temperature",
             "description": "Sampling temperature", "default": 1.0,
             "min": 0.1, "max": 2.0, "step": 0.1, "category": "Generation"},
            {"id": "cfg_scale", "type": "float", "label": "CFG Scale",
             "description": "Classifier-free guidance scale", "default": 1.5,
             "min": 0.1, "max": 10.0, "step": 0.1, "category": "Generation"},
            {"id": "lazy_load", "type": "bool", "label": "Lazy Loading",
             "description": "Load modules on demand to save GPU memory",
             "default": False, "category": "Performance"},
        ]

    @classmethod
    def get_prompt_schema(cls) -> PromptSchema:
        return {
            "required": True,
            "title": "📝 Lyrics & Tags",
            "description": "Enter lyrics with structure tags and style tags separated by commas",
            "placeholder": "[Verse]\nLyrics here...\n\nTags: piano,happy,wedding"
        }

    @classmethod
    def get_examples(cls) -> List[dict]:
        return [
            {
                "name": "Epic Orchestral",
                "prompt": "[Verse]\nThe mountains rise, the valleys fall\nThe eagle soars above it all\n\n[Chorus]\nWe are the champions of the light\n\nTags: orchestral, epic, cinematic, strings, brass, powerful",
                "params": {"temperature": 1.1, "cfg_scale": 1.5, "topk": 50}
            },
            {
                "name": "Lo-Fi Chill",
                "prompt": "[Verse]\nRaindrops on the windowpane\nSipping tea, ignoring the rain\n\nTags: lofi, chillhop, mellow, piano, vinyl crackle, relaxing",
                "params": {"temperature": 0.9, "cfg_scale": 1.2, "topk": 40}
            },
            {
                "name": "Upbeat Pop",
                "prompt": "[Verse]\nDancing in the summer sun\nHaving fun with everyone\n\nTags: pop, upbeat, synth, bright, female vocals, catchy",
                "params": {"temperature": 1.2, "cfg_scale": 2.0, "topk": 60}
            }
        ]

    @classmethod
    def get_metadata(cls) -> BackendMetadata:
        return {
            "name": "HeartMuLa Backend",
            "description": "Open-source music generation with lyrics and tags conditioning",
            "details": "Generates music from lyrics and style tags using HeartMuLa",
            "icon": "🎸",
            "requirements": "Python 3.10, PyTorch, CUDA GPU (8GB+ VRAM)"
        }

    def _load_model(self, config) -> None:
        """Load HeartMuLa model configuration."""
        backend_settings = {}
        if hasattr(config, 'get_backend_settings'):
            backend_settings = config.get_backend_settings("heartmula") or {}

        global_heartlib = getattr(config, 'heartlib_repo_dir', '') or ""
        global_ckpt = getattr(config, 'ckpt_dir', '') or ""

        self.heartlib_path = (
            backend_settings.get("heartlib_path") 
            or global_heartlib 
            or "./heartlib"
        )
        self.model_path = (
            backend_settings.get("model_path") 
            or global_ckpt 
            or "./ckpt"
        )
        self.version = backend_settings.get("version", "3B")

        self.heartlib_path = os.path.expanduser(str(self.heartlib_path).strip())
        self.model_path = os.path.expanduser(str(self.model_path).strip())

        if not os.path.exists(self.heartlib_path):
            raise BackendLoadError(
                f"HeartLib path not found: {self.heartlib_path}\n"
                "Fix: Set the 'HeartLib Path' in the Parameters menu to the repository folder."
            )
        if not os.path.exists(self.model_path):
            raise BackendLoadError(
                f"Model path not found: {self.model_path}\n"
                "Fix: Set the 'Model Checkpoint Path' in the Parameters menu to your models folder."
            )

        if not os.listdir(self.model_path):
            raise BackendLoadError(
                f"Model checkpoint directory is empty: {self.model_path}\n"
                "Please ensure your model weights are downloaded and placed in this folder."
            )

        script = os.path.join(self.heartlib_path, "examples", "run_music_generation.py")
        if not os.path.exists(script):
            raise BackendLoadError(
                f"Expected script not found: {script}\n"
                f"heartlib_path must point to the repo root (must contain the 'examples/' folder)"
            )

        logger.info("HeartMuLa loaded: heartlib=%s, model=%s, version=%s",
                    self.heartlib_path, self.model_path, self.version)

    def _generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        progress_callback: Optional[ProgressCallback],
        stop_check: Optional[StopCheck]
    ) -> Optional[GenerationResult]:
        """Generate music using HeartMuLa CLI."""

        tags_content = params.get("tags", "")
        prompt_content = prompt
        
        if not tags_content and "Tags:" in prompt:
            parts = prompt.split("Tags:", 1)
            prompt_content = parts[0].strip()
            tags_content = parts[1].strip()

        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt', encoding='utf-8'
        ) as lyrics_file:
            lyrics_file.write(prompt_content)
            lyrics_path = lyrics_file.name

        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt', encoding='utf-8'
        ) as tags_file:
            tags_file.write(tags_content)
            tags_path = tags_file.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name

        try:
            cmd = [
                sys.executable,
                os.path.join(self.heartlib_path, "examples", "run_music_generation.py"),
                "--model_path", self.model_path,
                "--version", self.version,
                "--lyrics", lyrics_path,
                "--tags", tags_path,
                "--save_path", output_path,
                "--max_audio_length_ms", str(duration * 1000),
                "--topk", str(params.get("topk", 50)),
                "--temperature", str(params.get("temperature", 1.0)),
                "--cfg_scale", str(params.get("cfg_scale", 1.5))
            ]

            if params.get("lazy_load", False):
                cmd += ["--lazy_load", "true"]

            logger.info("Running: %s", " ".join(cmd))

            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "cwd": self.heartlib_path,
                "encoding": 'utf-8',
                "errors": 'replace',
            }
            
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(cmd, **popen_kwargs)

            while True:
                if stop_check and stop_check():
                    self._process.terminate()
                    raise BackendCancelled("Generation cancelled by user")

                output = self._process.stdout.readline()
                
                if output == '' and self._process.poll() is not None:
                    break

                if output:
                    line = output.strip()
                    logger.debug(line)
                    if progress_callback and "%" in line:
                        try:
                            percent = int(line.split("%")[0].split()[-1])
                            progress_callback(percent, "Generating music...")
                        except (ValueError, IndexError):
                            pass
                else:
                    time.sleep(0.1)

            if self._process.returncode != 0:
                error_output = self._process.stderr.read()
                logger.error("HeartMuLa stderr:\n%s", error_output)
                
                err_lower = error_output.lower()
                if "out of memory" in err_lower:
                    raise BackendGenerateError(
                        "GPU ran out of VRAM (CUDA OOM).\n\n"
                        "Try reducing the 'Duration', enabling 'Lazy Loading' in Parameters, "
                        "or switching to a smaller model version."
                    )
                elif "no module named" in err_lower:
                    raise BackendGenerateError(
                        f"Missing Python dependency in HeartLib environment:\n{error_output}\n\n"
                        "Did you install requirements.txt inside the HeartLib repo folder?"
                    )
                else:
                    raise BackendGenerateError(
                        f"HeartMuLa generation failed (exit {self._process.returncode}):\n{error_output}"
                    )

            audio, sr = librosa.load(output_path, sr=44100)
            return GenerationResult(audio=audio, sample_rate=sr)

        finally:
            for f in (lyrics_path, tags_path, output_path):
                try:
                    os.unlink(f)
                except OSError:
                    pass
            self._process = None

    def terminate(self) -> None:
        """Forcefully terminate the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            except Exception as e:
                logger.error("Error terminating process: %s", e)
            self._process = None

    def _unload(self) -> None:
        """Unload model and cleanup resources."""
        self.terminate()
        logger.info("HeartMuLa backend unloaded")