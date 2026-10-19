"""Environment checks shared by `clipforge doctor` and the GUI."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import Config


@dataclass
class Check:
    name: str
    ok: bool
    hint: str = ""


def run_checks(cfg: Config) -> List[Check]:
    from .media import has_encoder, run

    checks: List[Check] = []

    def check(name: str, ok: bool, hint: str = "") -> None:
        checks.append(Check(name, ok, "" if ok else hint))

    has_ffmpeg = bool(shutil.which("ffmpeg"))
    check("ffmpeg в PATH", has_ffmpeg, "winget install Gyan.FFmpeg, затем перезапустите терминал / программу")
    check("ffprobe в PATH", bool(shutil.which("ffprobe")), "ставится вместе с ffmpeg (Gyan.FFmpeg)")
    if has_ffmpeg:
        filters = run(["ffmpeg", "-hide_banner", "-filters"])
        check("ffmpeg с libass (субтитры)", " ass " in filters, "нужна полная сборка ffmpeg: winget install Gyan.FFmpeg")
        if cfg.render.encoder.endswith("_nvenc"):
            nv = has_encoder(cfg.render.encoder)
            hint = "обновите драйвер NVIDIA; пока будет использоваться libx264 (медленнее)"
            if nv:
                try:
                    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                         "testsrc2=size=1280x720:rate=30:duration=1", "-c:v", cfg.render.encoder, "-f", "null", "-"])
                except RuntimeError as e:
                    nv = False
                    hint += f" ({str(e).splitlines()[-1]})"
            check(f"Кодировщик {cfg.render.encoder}", nv, hint)

    try:
        from .transcribe import add_cuda_dll_paths

        add_cuda_dll_paths()
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        check(f"CUDA для Whisper (устройств: {n})", n > 0 or cfg.whisper.device == "cpu",
              "pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\" и обновите драйвер NVIDIA")
    except Exception as e:
        check("faster-whisper установлен", False, f"pip install -r requirements.txt ({e})")

    try:
        import cv2

        check("OpenCV с детектором лиц", hasattr(cv2, "FaceDetectorYN"), "pip install -U opencv-python")
    except ImportError:
        check("OpenCV установлен", False, "pip install -r requirements.txt")

    llm = cfg.llm
    if llm.server_cmd:
        cmd = llm.server_cmd.strip()
        exe = cmd.split('"')[1] if cmd.startswith('"') else cmd.split()[0]
        check(f"llama-server: {exe}", Path(exe).exists() or bool(shutil.which(exe)),
              "скачайте llama.cpp (сборка win-cuda) и поправьте путь")
    elif llm.server_exe:
        check(f"llama-server: {llm.server_exe}", Path(llm.server_exe).exists() or bool(shutil.which(llm.server_exe)),
              "скачайте llama.cpp (сборка win-cuda) и укажите путь к llama-server.exe")
        check(f"Модель: {llm.model_path or '(не указана)'}", bool(llm.model_path) and Path(llm.model_path).is_file(),
              "скачайте .gguf модель и укажите путь к ней")
    else:
        import requests

        try:
            r = requests.get(llm.base_url.rstrip("/") + "/models", timeout=5,
                             headers={"Authorization": f"Bearer {llm.api_key}"} if llm.api_key else {})
            check(f"LLM сервер {llm.base_url}", r.ok, r.text[:200])
        except requests.RequestException as e:
            check(f"LLM сервер {llm.base_url}", False, f"запустите сервер или укажите путь к llama-server ({e})")
    return checks
