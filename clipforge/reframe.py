"""Face-aware 16:9 -> 9:16 reframing: detect faces and cuts, then plan a smooth virtual camera."""
from __future__ import annotations

import logging
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ReframeCfg
from .media import NO_WINDOW, VideoInfo

log = logging.getLogger(__name__)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def ensure_face_model(cache_dir: Path) -> Path:
    path = cache_dir / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading face detector model -> %s", path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        tmp.replace(path)
    return path


def crop_width_norm(info: VideoInfo, cfg: ReframeCfg) -> float:
    """Width of the vertical crop as a fraction of the source width (1.0 = no horizontal crop)."""
    target = cfg.output_width / cfg.output_height
    return min(1.0, info.height * target / info.width)


def _signature(frame: np.ndarray):
    import cv2

    hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    return hist, thumb.astype(np.float32)


def _cut_score(a, b, cfg: ReframeCfg) -> float:
    """>= 1.0 means a scene cut between two frame signatures."""
    import cv2

    hist_d = cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA) / cfg.cut_threshold
    pix_d = float(np.mean(np.abs(a[1] - b[1]))) / cfg.cut_pixel_threshold
    return max(hist_d, pix_d)


def analyze(video: Path, start: float, end: float, info: VideoInfo, cfg: ReframeCfg, model: Path) -> dict:
    """Decode the clip at a low fps and collect face boxes (normalized) and scene cuts."""
    import cv2

    dw = cfg.detect_width
    dh = int(round(dw * info.height / info.width / 2)) * 2
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video),
        "-t", f"{end - start:.3f}", "-vf", f"fps={cfg.analysis_fps},scale={dw}:{dh}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    detector = cv2.FaceDetectorYN.create(str(model), "", (dw, dh), cfg.min_face_conf, 0.3, 50)
    frame_bytes = dw * dh * 3
    times: List[float] = []
    faces: List[List[Tuple[float, float, float, float, float]]] = []
    cuts: List[float] = []
    prev_sig = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    try:
        i = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(dh, dw, 3)
            t = i / cfg.analysis_fps
            sig = _signature(frame)
            if prev_sig is not None and _cut_score(prev_sig, sig, cfg) >= 1.0:
                cuts.append(t)
            prev_sig = sig
            _, det = detector.detect(frame)
            found = []
            for row in det if det is not None else []:
                x, y, w, h, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[-1])
                if w / dw < cfg.min_face_size:
                    continue
                found.append(((x + w / 2) / dw, (y + h / 2) / dh, w / dw, h / dh, score))
            times.append(t)
            faces.append(found)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
    cuts = [_refine_cut(video, start, c, info, cfg) for c in cuts]
    return {"times": times, "faces": faces, "cuts": cuts}


def _refine_cut(video: Path, start: float, t: float, info: VideoInfo, cfg: ReframeCfg) -> float:
    """A cut was seen between two analysis samples; find the exact frame at the source frame rate."""
    t0 = max(0.0, t - 1.0 / cfg.analysis_fps)
    w, h = 160, 90
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start + t0:.3f}", "-i", str(video),
        "-t", f"{t - t0 + 0.05:.3f}", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW).stdout
    except subprocess.TimeoutExpired:
        return t
    n = len(raw) // (w * h * 3)
    if n < 2:
        return t
    frames = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    sigs = [_signature(f) for f in frames]
    diffs = [_cut_score(sigs[i - 1], sigs[i], cfg) for i in range(1, n)]
    best = int(np.argmax(diffs)) + 1
    return round(t0 + best / info.fps, 4)


def _tracks(faces: List[list], max_jump: float = 0.12) -> List[dict]:
    """Greedy frame-to-frame association of face detections into tracks."""
    tracks: List[dict] = []
    active: List[dict] = []
    for i, dets in enumerate(faces):
        used = set()
        next_active = []
        for tr in active:
            last = tr["pts"][-1]
            best, best_d = None, max_jump
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dist = abs(d[0] - last[1]) + abs(d[1] - last[2])
                if dist < best_d:
                    best, best_d = j, dist
            if best is not None:
                used.add(best)
                d = dets[best]
                tr["pts"].append((i, d[0], d[1], d[2], d[4]))
                next_active.append(tr)
            elif i - last[0] <= 4:  # tolerate short detection gaps
                next_active.append(tr)
        for j, d in enumerate(dets):
            if j not in used:
                tr = {"pts": [(i, d[0], d[1], d[2], d[4])]}
                tracks.append(tr)
                next_active.append(tr)
        active = next_active
    for tr in tracks:
        tr["weight"] = sum(p[3] * p[4] for p in tr["pts"])
    return sorted(tracks, key=lambda t: t["weight"], reverse=True)


def _series(track: dict, n: int, offset: int) -> np.ndarray:
    """Per-sample x center of a track, NaN where it is absent."""
    xs = np.full(n, np.nan)
    for idx, x, *_ in track["pts"]:
        xs[idx - offset] = x
    return xs


def _fill(xs: np.ndarray, default: float) -> np.ndarray:
    ok = ~np.isnan(xs)
    if not ok.any():
        return np.full_like(xs, default)
    idx = np.arange(len(xs))
    return np.interp(idx, idx[ok], xs[ok])


def _follow(target: np.ndarray, fps: float, cfg: ReframeCfg) -> np.ndarray:
    """Deadzone + speed-limited follow, then a zero-phase moving average."""
    if np.ptp(target) < cfg.lock_threshold:
        return np.full_like(target, float(np.median(target)))
    max_step = cfg.max_speed / fps
    out = np.empty_like(target)
    x = float(np.median(target[: max(1, int(fps))]))
    moving = False
    for i, t in enumerate(target):
        diff = t - x
        # Hysteresis: start panning only when the face leaves the deadzone,
        # then keep going until it is centered again.
        if abs(diff) > cfg.deadzone:
            moving = True
        elif abs(diff) < cfg.deadzone * 0.15:
            moving = False
        if moving:
            x += float(np.clip(diff * 0.35, -max_step, max_step))
        out[i] = x
    k = max(1, int(cfg.smooth_seconds * fps)) | 1
    if k > 1 and len(out) > k:
        pad = k // 2
        padded = np.pad(out, pad, mode="edge")
        out = np.convolve(padded, np.ones(k) / k, mode="valid")
    return out


def plan_camera(analysis: dict, crop_w: float, cfg: ReframeCfg) -> List[Tuple[float, float]]:
    """Returns [(time, crop_center_x_normalized)] keyframes for linear interpolation.
    The camera moves smoothly inside a shot and jumps only on scene cuts."""
    times, faces = analysis["times"], analysis["faces"]
    n = len(times)
    if n == 0:
        return [(0.0, 0.5)]
    fps = cfg.analysis_fps
    cut_at = {}
    for c in analysis["cuts"]:
        idx = int(np.ceil(c * fps - 1e-6))
        if 0 < idx < n:
            cut_at[idx] = c
    bounds = [0] + sorted(cut_at) + [n]
    half = crop_w / 2
    xs = np.empty(n)
    prev_center = 0.5
    for a, b in zip(bounds[:-1], bounds[1:]):
        tracks = [t for t in _tracks(faces[a:b]) if len(t["pts"]) >= max(2, 0.15 * (b - a))]
        if not tracks:
            target = np.full(b - a, prev_center)
        else:
            main = _fill(_series(tracks[0], b - a, 0), prev_center)
            target = main
            if cfg.two_shot and len(tracks) > 1 and tracks[1]["weight"] > 0.5 * tracks[0]["weight"]:
                second = _fill(_series(tracks[1], b - a, 0), float(np.median(main)))
                spread = np.abs(main - second) + 2 * max(p[3] for p in tracks[0]["pts"])
                if np.percentile(spread, 90) < crop_w * 0.95:
                    target = (main + second) / 2
        xs[a:b] = np.clip(_follow(target, fps, cfg), half, 1 - half)
        prev_center = float(xs[b - 1])

    keys: List[Tuple[float, float]] = []
    for i, (t, x) in enumerate(zip(times, xs)):
        if i in cut_at:
            c = cut_at[i]
            keys.append((c - 0.001, float(xs[i - 1])))  # hold the old framing until the cut frame
            keys.append((c, float(x)))
        keys.append((float(t), float(x)))
    return keys


def reframe(video: Path, start: float, end: float, info: VideoInfo, cfg: ReframeCfg,
            model: Path) -> Optional[List[Tuple[float, float]]]:
    """Returns camera keyframes [(t, center_x_norm)], or None if no horizontal crop is needed."""
    crop_w = crop_width_norm(info, cfg)
    if crop_w >= 0.999:
        return None
    analysis = analyze(video, start, end, info, cfg, model)
    with_face = sum(1 for f in analysis["faces"] if f)
    log.info("  faces in %d/%d sampled frames, %d cuts", with_face, len(analysis["faces"]), len(analysis["cuts"]))
    return plan_camera(analysis, crop_w, cfg)
