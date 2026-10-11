# acestep_backend.py
import logging
from typing import Optional, List
import numpy as np
import torch
from .base import (
    MusicBackend, SettingItem, PromptSchema, BackendMetadata,
    GenerationResult, ProgressCallback, StopCheck,
    BackendError, BackendLoadError, BackendGenerateError, BackendCancelled,
    BackendState, BackendCapabilities,
)

logger = logging.getLogger(__name__)

class ACEStepBackend(MusicBackend):
    """ACE-Step music generation backend."""
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
    
    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            supports_seed=True,
            supports_batch=False,
            supports_streaming=False,
            min_vram_gb=6.0,
            max_duration_sec=300,  # 5 minutes
            output_sample_rates=[44100, 48000],
            description="ACE-Step foundation model for fast, high-quality music generation"
        )
    
    @classmethod
    def get_settings_schema(cls) -> List[SettingItem]:
        return [
            {
                "id": "model_name",
                "type": "select",
                "label": "Model Version",
                "description": "ACE-Step model version",
                "default": "ACE-Step/ACE-Step-v1-3.5B",
                "options": [
                    "ACE-Step/ACE-Step-v1-3.5B",
                    "ACE-Step/ACE-Step-v1.5",
                    "ACE-Step/ACE-Step-v1.5-xl"
                ],
                "category": "Model"
            },
            {
                "id": "device",
                "type": "select",
                "label": "Device",
                "description": "Device to run model on",
                "default": "cuda",
                "options": ["cuda", "cpu"],
                "category": "Performance"
            },
            {
                "id": "torch_dtype",
                "type": "select",
                "label": "Precision",
                "description": "Model precision",
                "default": "float16",
                "options": ["float16", "float32", "bfloat16"],
                "category": "Performance"
            },
            {
                "id": "guidance_scale",
                "type": "float",
                "label": "Guidance Scale",
                "description": "How closely to follow the prompt (1.0-10.0)",
                "default": 4.5,
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
                "category": "Generation"
            },
            {
                "id": "manual_seed",
                "type": "int",
                "label": "Random Seed",
                "description": "Fixed seed for reproducibility (0 for random)",
                "default": 42,
                "min": 0,
                "max": 1000000,
                "category": "Generation"
            }
        ]
    
    @classmethod
    def get_prompt_schema(cls) -> PromptSchema:
        return {
            "required": True,
            "title": "📝 Prompt & Lyrics",
            "description": "Enter musical style prompt and lyrics with structure tags",
            "placeholder": "Prompt: Jazz piano trio, upright bass, brushed drums\n\nLyrics:\n[verse]\nLyrics here..."
        }
    
    @classmethod
    def get_metadata(cls) -> BackendMetadata:
        return {
            "name": "ACE-Step Backend",
            "description": "Fast, high-quality music generation foundation model",
            "details": "Generates music up to 4 minutes in 20 seconds with superior musical coherence",
            "icon": "⚡",
            "requirements": "Python 3.10+, PyTorch, CUDA GPU (6GB+ VRAM)"
        }
    
    def _load_model(self, config: dict) -> None:
        """Load ACE-Step model."""
        try:
            # Import here to avoid loading if not used
            from acestep.pipeline_ace_step import ACEStepPipeline
            
            model_name = config.get("model_name", "ACE-Step/ACE-Step-v1-3.5B")
            self.device = config.get("device", "cuda")
            dtype_str = config.get("torch_dtype", "float16")
            
            # Convert dtype string to torch dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16
            }
            self.torch_dtype = dtype_map.get(dtype_str, torch.float16)
            
            logger.info(f"Loading ACE-Step model: {model_name}")
            
            # Load pipeline
            self.pipeline = ACEStepPipeline.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device=self.device
            )
            
            logger.info("ACE-Step model loaded successfully")
            
        except ImportError as e:
            raise BackendLoadError(f"ACE-Step not installed: {e}. Install with: pip install git+https://github.com/ace-step/ACE-Step.git")
        except Exception as e:
            raise BackendLoadError(f"Failed to load ACE-Step model: {e}")
    
    def _generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        progress_callback: Optional[ProgressCallback],
        stop_check: Optional[StopCheck]
    ) -> Optional[GenerationResult]:
        """Generate music using ACE-Step API."""
        
        if not self.pipeline:
            raise BackendGenerateError("Pipeline not loaded")
        
        try:
            # Parse prompt format: "Prompt: ...\n\nLyrics:\n..."
            if "\n\n" in prompt:
                parts = prompt.split("\n\n", 1)
                prompt_text = parts[0].replace("Prompt: ", "").strip()
                lyrics = parts[1].replace("Lyrics:", "").strip()
            else:
                prompt_text = prompt
                lyrics = "[inst]"  # Default to instrumental
            
            # Generate audio
            if progress_callback:
                progress_callback(10, "Starting generation...")
            
            # Check for cancellation
            if stop_check and stop_check():
                raise BackendCancelled("Generation cancelled by user")
            
            audio_array = self.pipeline(
                prompt=prompt_text,
                lyrics=lyrics,
                audio_duration=duration,
                guidance_scale=params.get("guidance_scale", 4.5),
                manual_seed=params.get("manual_seed", 42)
            )
            
            if progress_callback:
                progress_callback(90, "Finalizing audio...")
            
            # Convert to numpy array if needed
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            # Get sample rate from pipeline config
            sample_rate = self.pipeline.config.sampling_rate
            
            return GenerationResult(
                audio=audio_array,
                sample_rate=sample_rate
            )
            
        except Exception as e:
            raise BackendGenerateError(f"ACE-Step generation failed: {e}")
    
    def _unload(self) -> None:
        """Unload model and free resources."""
        if self.pipeline:
            # Move model to CPU to free GPU memory
            if hasattr(self.pipeline, 'to'):
                self.pipeline.to('cpu')
            self.pipeline = None
            torch.cuda.empty_cache()
            logger.info("ACE-Step backend unloaded")