"""Mathematical synthesis backend — no AI models required."""
import logging
import numpy as np
from typing import Optional, List

from .base import (
    MusicBackend, SettingItem, PromptSchema, BackendMetadata,
    GenerationResult, ProgressCallback, StopCheck,
    BackendError, BackendGenerateError, BackendCancelled,
    BackendCapabilities, AudioUtils,
)

logger = logging.getLogger(__name__)

# Musical scales (semitone offsets from root)
SCALES = {
    "pentatonic": [0, 3, 5, 7, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


class MathSynthBackend(MusicBackend):
    """Algorithmic music generation using mathematical synthesis."""

    def __init__(self):
        super().__init__()
        self.sample_rate = 44100

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            supports_seed=True,
            supports_batch=False,
            supports_streaming=True,
            min_vram_gb=0.0,
            max_duration_sec=300,
            output_sample_rates=[44100],
            description="Lissajous curves, Fibonacci sequences, mathematical synthesis",
        )

    @classmethod
    def get_settings_schema(cls) -> List[SettingItem]:
        return [
            {
                "id": "pattern",
                "type": "dropdown",
                "label": "Pattern",
                "description": "Algorithm to use for note/frequency generation",
                "default": "lissajous",
                "options": ["lissajous", "fibonacci", "sine_chord"],
                "category": "Synthesis",
            },
            {
                "id": "base_frequency",
                "type": "slider",
                "label": "Base Frequency",
                "description": "Root frequency for the synthesis (Hz)",
                "default": 220.0,
                "min": 20.0,
                "max": 2000.0,
                "step": 1.0,
                "category": "Synthesis",
            },
            {
                "id": "scale",
                "type": "dropdown",
                "label": "Musical Scale",
                "description": "Scale constraint for sequence generation",
                "default": "pentatonic",
                "options": ["pentatonic", "major", "minor", "chromatic"],
                "category": "Music Theory",
            },
            {
                "id": "waveform",
                "type": "dropdown",
                "label": "Waveform",
                "description": "Base oscillator shape",
                "default": "sine",
                "options": ["sine", "square", "sawtooth", "triangle"],
                "category": "Synthesis",
            },
            {
                "id": "a_b_ratio",
                "type": "text",
                "label": "Lissajous a:b Ratio",
                "description": "Frequency ratio for Lissajous stereo modulation (e.g. '3:2')",
                "default": "3:2",
                "category": "Synthesis",
            },
        ]

    @classmethod
    def get_prompt_schema(cls) -> PromptSchema:
        return {
            "required": False,
            "title": "📝 Custom Parameters (Optional)",
            "description": "Override base frequency or ratio (e.g. '440 5:4') or leave blank to use settings",
            "placeholder": "440 5:4",
        }

    @classmethod
    def get_metadata(cls) -> BackendMetadata:
        return {
            "name": "Math Synth",
            "description": "Algorithmic music generation using pure mathematics",
            "details": "No AI models required. Generates audio via Lissajous curves and Fibonacci sequences.",
            "icon": "🧮",
            "requirements": "NumPy, CPU only",
        }

    def _load_model(self, config) -> None:
        # Support both Config objects and plain dicts
        if hasattr(config, "get"):
            self.sample_rate = config.get("sample_rate", 44100)
        elif isinstance(config, dict):
            self.sample_rate = config.get("sample_rate", 44100)
        logger.info("Math Synth backend initialized (sr=%d)", self.sample_rate)

    # ── Waveform generation ──
    def _generate_waveform(self, t: np.ndarray, freq: float, waveform: str) -> np.ndarray:
        phase = 2 * np.pi * freq * t
        if waveform == "square":
            return np.sign(np.sin(phase))
        if waveform == "sawtooth":
            return 2.0 * (t * freq - np.floor(0.5 + t * freq))
        if waveform == "triangle":
            return 2.0 * np.abs(2.0 * (t * freq - np.floor(0.5 + t * freq))) - 1.0
        return np.sin(phase)  # default: sine

    def _apply_envelope(self, signal: np.ndarray, attack: float = 0.01,
                        release: float = 0.1) -> np.ndarray:
        sr = self.sample_rate
        env = np.ones_like(signal)
        attack_samples = int(sr * attack)
        release_samples = int(sr * release)
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            env[-release_samples:] = np.linspace(1, 0, release_samples)
        return signal * env

    @staticmethod
    def _parse_ratio(ratio_str: str, default=(3, 2)):
        try:
            parts = ratio_str.split(":")
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                if a > 0 and b > 0:
                    return a, b
        except (ValueError, AttributeError):
            pass
        return default

    # ── Pattern generators ──
    def _generate_lissajous(self, duration: int, params: dict) -> np.ndarray:
        sr = self.sample_rate
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

        base_freq = float(params.get("base_frequency", 220.0))
        ratio_str = str(params.get("a_b_ratio", "3:2"))
        waveform = params.get("waveform", "sine")
        a, b = self._parse_ratio(ratio_str)

        freq_a = base_freq
        freq_b = base_freq * (b / a)

        left = self._generate_waveform(t, freq_a, waveform)
        right = self._generate_waveform(t, freq_b, waveform)

        # True phase shift (π/2) instead of np.roll (which causes wrap artifacts)
        right_shifted = self._generate_waveform(t - 0.25 / freq_b, freq_b, waveform)

        left = self._apply_envelope(left, attack=0.5, release=0.5)
        right_shifted = self._apply_envelope(right_shifted, attack=0.5, release=0.5)

        left = AudioUtils.safe_normalize(left, 0.8)
        right_shifted = AudioUtils.safe_normalize(right_shifted, 0.8)

        return np.stack([left, right_shifted], axis=0)  # (2, n)

    def _generate_fibonacci(self, duration: int, params: dict, seed: int = 0) -> np.ndarray:
        sr = self.sample_rate
        base_freq = float(params.get("base_frequency", 220.0))
        scale_type = params.get("scale", "pentatonic")
        waveform = params.get("waveform", "sine")
        scale = SCALES.get(scale_type, SCALES["pentatonic"])

        # Generate Fibonacci sequence with seed offset
        fib = [0, 1]
        while len(fib) < 128:
            fib.append(fib[-1] + fib[-2])
        # Apply seed as offset into sequence
        offset = seed % 64 if seed else 0

        t_step = 0.2  # 200ms per note
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        num_notes = int(duration / t_step)

        for i in range(num_notes):
            fib_val = fib[(i + offset) % len(fib)]
            note_idx = fib_val % len(scale)
            semitones = scale[note_idx]
            freq = base_freq * (2 ** (semitones / 12))

            t_note = np.linspace(0, t_step, int(sr * t_step), endpoint=False, dtype=np.float32)
            note = self._generate_waveform(t_note, freq, waveform)
            note = self._apply_envelope(note, attack=0.01, release=0.05)

            start = i * int(sr * t_step)
            end = start + int(sr * t_step)
            if end <= len(audio):
                audio[start:end] += note

        audio = AudioUtils.safe_normalize(audio, 0.8)
        return np.stack([audio, audio], axis=0)

    def _generate_sine_chord(self, duration: int, params: dict) -> np.ndarray:
        sr = self.sample_rate
        t_step = 0.5  # 500ms per chord
        num_chords = int(duration / t_step)
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        
        # I - V - vi - IV progression (relative to base freq)
        progressions = [
            [1, 5/4, 3/2],       # I
            [3/2, 2, 5/3],       # V
            [9/8, 4/3, 5/3],     # vi
            [4/3, 5/3, 2],       # IV
        ]
        
        for i in range(num_chords):
            chord_ratios = progressions[i % len(progressions)]
            base = float(params.get("base_frequency", 220.0))
            waveform = params.get("waveform", "sine")
            
            t_note = np.linspace(0, t_step, int(sr * t_step), endpoint=False, dtype=np.float32)
            chord = sum(self._generate_waveform(t_note, base * r, waveform) for r in chord_ratios) / len(chord_ratios)
            chord = self._apply_envelope(chord, attack=0.05, release=0.1)
            
            start = i * int(sr * t_step)
            end = start + int(sr * t_step)
            if end <= len(audio):
                audio[start:end] += chord
                
        audio = AudioUtils.safe_normalize(audio, 0.85)
        return np.stack([audio, audio], axis=0)
    
    # ── Main generate ──
    def _generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        progress_callback: Optional[ProgressCallback],
        stop_check: Optional[StopCheck],
    ) -> Optional[GenerationResult]:
        try:
            # Work on a copy so we don't mutate caller's params
            p = dict(params)

            # Parse prompt overrides: "<freq> <ratio>"
            prompt = prompt.strip()
            if prompt:
                parts = prompt.split()
                if len(parts) >= 1:
                    try:
                        p["base_frequency"] = float(parts[0])
                    except ValueError:
                        pass
                if len(parts) >= 2:
                    p["a_b_ratio"] = parts[1]

            if progress_callback:
                progress_callback(10, "Initializing synthesis…")

            pattern = p.get("pattern", "lissajous")
            seed = int(p.get("seed", 0))

            if stop_check and stop_check():
                raise BackendCancelled("Generation cancelled by user")

            if progress_callback:
                progress_callback(30, f"Generating {pattern} pattern…")

            if pattern == "lissajous":
                audio = self._generate_lissajous(duration, p)
            elif pattern == "fibonacci":
                audio = self._generate_fibonacci(duration, p, seed=seed)
            elif pattern == "sine_chord":
                audio = self._generate_sine_chord(duration, p)
            else:
                raise BackendGenerateError(f"Unknown pattern: {pattern}")

            if stop_check and stop_check():
                raise BackendCancelled("Generation cancelled by user")

            if progress_callback:
                progress_callback(90, "Finalizing audio…")

            audio = AudioUtils.safe_normalize(audio, 0.9)

            return GenerationResult(
                audio=audio.astype(np.float32),
                sample_rate=self.sample_rate,
            )

        except BackendCancelled:
            raise
        except BackendError:
            raise
        except Exception as e:
            raise BackendGenerateError(f"Math synthesis failed: {e}") from e

    def _unload(self) -> None:
        pass