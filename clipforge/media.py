"""ffmpeg / ffprobe / yt-dlp helpers."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Hide console windows of child processes on Windows (matters when running from the GUI).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class VideoInfo:
    width: int  # display width (rotation applied)
    height: int
    fps: float
    duration: float


def run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    log.debug("run: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=NO_WINDOW,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd[0]}\n{tail}")
    return proc.stdout


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH (Windows: winget install Gyan.FFmpeg)")


@lru_cache(maxsize=None)
def has_encoder(name: str) -> bool:
    try:
        out = run(["ffmpeg", "-hide_banner", "-encoders"])
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def _parse_rate(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(rate or 0)


def probe(path: Path) -> VideoInfo:
    if shutil.which("ffprobe"):
        out = run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ])
        data = json.loads(out)
        st = data["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        rotation = 0
        for sd in st.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = int(sd["rotation"])
        rotation = int(st.get("tags", {}).get("rotate", rotation))
        if abs(rotation) % 180 == 90:
            w, h = h, w
        fps = _parse_rate(st.get("avg_frame_rate", "0")) or _parse_rate(st.get("r_frame_rate", "30"))
        duration = float(data["format"].get("duration") or st.get("duration") or 0)
        return VideoInfo(w, h, fps or 30.0, duration)

    # Fallback without ffprobe (rotation metadata is not handled here).
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    info = VideoInfo(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, frames / fps
    )
    cap.release()
    return info


def download(url: str, dest_dir: Path) -> Path:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        candidates = sorted(dest_dir.glob(f"{info['id']}.*"))
        if not candidates:
            raise RuntimeError(f"Download finished but file not found for {url}")
        path = candidates[0]
    return path


def extract_audio(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
