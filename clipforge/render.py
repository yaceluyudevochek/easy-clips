"""Final encode: moving crop (via sendcmd) + scale + burned-in captions + audio."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ReframeCfg, RenderCfg
from .media import VideoInfo, has_encoder, run

log = logging.getLogger(__name__)


def crop_commands(keys: List[Tuple[float, float]], info: VideoInfo, crop_px: int, duration: float) -> Tuple[int, str]:
    """Interpolate camera keys per output frame; returns (initial x, sendcmd script)."""
    kt = np.array([k[0] for k in keys])
    kx = np.array([k[1] for k in keys])
    fps = min(info.fps, 60.0)
    n = int(np.ceil(duration * fps)) + 1
    t = np.arange(n) / fps
    centers = np.interp(t, kt, kx) * info.width
    xs = np.clip(np.round(centers - crop_px / 2), 0, info.width - crop_px).astype(int)
    xs -= xs % 2
    lines = []
    prev = int(xs[0])
    for ti, x in zip(t[1:], xs[1:]):
        if x != prev:
            lines.append(f"{ti:.4f} crop x {int(x)};")
            prev = int(x)
    return int(xs[0]), "\n".join(lines) + "\n"


def _encoder_args(cfg: RenderCfg) -> List[str]:
    enc = cfg.encoder
    if enc.endswith("_nvenc") and not has_encoder(enc):
        log.warning("%s not available in this ffmpeg build, using libx264", enc)
        enc = "libx264"
    if enc.endswith("_nvenc"):
        return ["-c:v", enc, "-preset", cfg.preset, "-tune", "hq", "-rc", "vbr", "-cq", str(cfg.cq), "-b:v", "0"]
    if enc == "libx264":
        return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cfg.cq - 1)]
    return ["-c:v", enc]


def render_clip(video: Path, info: VideoInfo, start: float, end: float, keys: Optional[List[Tuple[float, float]]],
                ass_text: Optional[str], out_path: Path, rcfg: ReframeCfg, cfg: RenderCfg) -> None:
    # ffmpeg runs with cwd = output dir so filter arguments are plain relative names
    # (absolute Windows paths like C:\... need painful escaping inside -vf).
    out_dir = out_path.parent
    stem = out_path.stem
    W, H = rcfg.output_width, rcfg.output_height
    duration = end - start
    filters = []
    if keys is not None:
        crop_px = int(round(info.height * W / H))
        crop_px -= crop_px % 2
        x0, script = crop_commands(keys, info, crop_px, duration)
        if script.strip():  # sendcmd fails on an empty file; a locked camera needs no commands
            (out_dir / f"{stem}.cmd").write_text(script, encoding="utf-8")
            filters.append(f"sendcmd=f={stem}.cmd")
        filters.append(f"crop=w={crop_px}:h={info.height}:x={x0}:y=0")
        filters.append(f"scale={W}:{H}:flags=lanczos")
    else:
        filters.append(f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos")
        filters.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black")
    filters.append("setsar=1")
    if ass_text:
        (out_dir / f"{stem}.ass").write_text(ass_text, encoding="utf-8")
        filters.append(f"ass={stem}.ass")
    if info.fps > 60:
        filters.append("fps=60")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{duration:.3f}",
        "-vf", ",".join(filters),
        *_encoder_args(cfg), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ar", "48000",
        "-movflags", "+faststart", out_path.name,
    ]
    try:
        try:
            run(cmd, cwd=out_dir)
        except RuntimeError:
            if not cfg.encoder.endswith("_nvenc"):
                raise
            log.warning("NVENC encode failed, retrying with libx264")
            cmd[cmd.index("-c:v"):cmd.index("-pix_fmt")] = ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]
            run(cmd, cwd=out_dir)
    finally:
        for suffix in (".cmd", ".ass"):
            (out_dir / f"{stem}{suffix}").unlink(missing_ok=True)
