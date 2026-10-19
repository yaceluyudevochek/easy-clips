"""Speech-to-text with word timestamps (faster-whisper)."""
from __future__ import annotations

import gc
import glob
import logging
import os
import site
import sys
from pathlib import Path

from .config import WhisperCfg
from .progress import Cancelled, Reporter

log = logging.getLogger(__name__)


def add_cuda_dll_paths() -> None:
    """On Windows, make DLLs from the nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels visible to CTranslate2."""
    if sys.platform != "win32":
        return
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass


def _transcribe(wav: Path, cfg: WhisperCfg, device: str, compute_type: str, report: Reporter) -> dict:
    from faster_whisper import WhisperModel

    log.info("Loading Whisper %s on %s (%s)", cfg.model, device, compute_type)
    model = WhisperModel(cfg.model, device=device, compute_type=compute_type)
    try:
        segments, info = model.transcribe(
            str(wav),
            language=cfg.language,
            beam_size=cfg.beam_size,
            word_timestamps=True,
            vad_filter=cfg.vad_filter,
        )
        log.info("Language: %s (p=%.2f), audio %.0f s", info.language, info.language_probability, info.duration)
        words = []
        last_report = 0.0
        for seg in segments:
            report("transcribe", seg.end, info.duration)
            for w in seg.words or []:
                words.append({"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)})
            if seg.end - last_report >= 300:
                last_report = seg.end
                log.info("  transcribed %.0f / %.0f s", seg.end, info.duration)
        return {"language": info.language, "duration": info.duration, "words": words}
    finally:
        del model
        gc.collect()


def transcribe(wav: Path, cfg: WhisperCfg, report: Reporter = Reporter()) -> dict:
    add_cuda_dll_paths()
    try:
        return _transcribe(wav, cfg, cfg.device, cfg.compute_type, report)
    except Cancelled:
        raise
    except Exception as e:
        if cfg.device == "cpu":
            raise
        log.warning("Whisper on %s failed (%s), falling back to CPU int8 — this will be slow", cfg.device, e)
        return _transcribe(wav, cfg, "cpu", "int8", report)
