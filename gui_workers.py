"""Background worker thread — supports batch generation with queue-aware progress."""
import logging
from dataclasses import dataclass
from typing import Optional, List

from PySide6.QtCore import QObject, Signal, Slot
from core import GenerationService

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    prompt: str
    duration: int
    params: dict
    output_dir: str
    variations: int = 1
    export_format: str = "WAV"
    organize_output: bool = True

class GenerationWorker(QObject):
    finished = Signal()
    success = Signal(str, dict)
    error = Signal(str)
    progress = Signal(int, str)
    queue_progress = Signal(int, int)

    def __init__(self, service: GenerationService, requests: List[GenerationRequest]):
        super().__init__()
        self._service = service
        self._requests = requests
        self._cancelled = False

    def stop(self):
        self._cancelled = True
        if hasattr(self._service._backend_svc, 'cancel_generation'):
            self._service._backend_svc.cancel_generation()

    @Slot()
    def run(self):
        total = len(self._requests)
        total_steps = sum(req.variations for req in self._requests)
        current_step = 0

        try:
            for idx, req in enumerate(self._requests):
                if self._cancelled:
                    break

                self.queue_progress.emit(idx + 1, total)

                for v in range(req.variations):
                    if self._cancelled:
                        break

                    base_progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0
                    step_fraction = 1.0 / total_steps if total_steps > 0 else 1.0

                    def local_progress(p: int, phase: str,
                                       _base=base_progress, _frac=step_fraction):
                        overall = _base + int((p / 100.0) * _frac * 100)
                        self.progress.emit(min(overall, 100), phase)

                    result = self._service.run_single(
                        prompt=req.prompt,
                        duration=req.duration,
                        params=req.params,
                        output_dir=req.output_dir,
                        variation=v + 1,
                        export_format=req.export_format,
                        organize_output=req.organize_output,
                        progress_callback=local_progress,
                        stop_check=lambda: self._cancelled,
                    )

                    current_step += 1

                    if self._cancelled:
                        break

                    if result:
                        self.success.emit(result["path"], result["analysis"])

            if not self._cancelled:
                self.progress.emit(100, "Complete")
        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()