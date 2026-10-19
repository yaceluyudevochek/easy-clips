"""Configuration: dataclasses with defaults, overridable from a YAML file."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class WhisperCfg:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: Optional[str] = None  # None = autodetect, "ru" to force
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class LLMCfg:
    # Any OpenAI-compatible endpoint: llama.cpp server, Ollama, LM Studio, cloud APIs.
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = ""
    model: str = "local"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 900
    json_schema: bool = True  # constrain output with response_format=json_schema
    # Local llama.cpp server started right before the LLM stage and stopped after it,
    # so Whisper and the LLM never share the 8 GB of VRAM. Leave server_exe empty for an external server.
    server_exe: str = ""  # path to llama-server(.exe)
    model_path: str = ""  # path to the .gguf model
    gpu_layers: int = 99
    n_cpu_moe: int = 48  # MoE layers whose experts stay in RAM (0 for dense models like Qwen3-8B)
    ctx_size: int = 16384
    extra_args: str = ""
    server_cmd: str = ""  # full command line; overrides the fields above when set
    server_ready_timeout: int = 600


@dataclass
class HighlightsCfg:
    min_duration: float = 20.0
    max_duration: float = 60.0
    max_clips: int = 8
    chunk_minutes: float = 8.0
    chunk_overlap_minutes: float = 1.5
    per_chunk: int = 4  # max candidates the LLM may return per chunk
    min_score: float = 6.0
    max_overlap: float = 0.3  # overlap / shorter clip above which candidates are duplicates
    rerank: bool = True
    rerank_top: int = 14


@dataclass
class ReframeCfg:
    output_width: int = 1080
    output_height: int = 1920
    analysis_fps: float = 8.0
    detect_width: int = 640
    min_face_conf: float = 0.7
    min_face_size: float = 0.04  # face width relative to frame width
    cut_threshold: float = 0.35  # Bhattacharyya distance between HSV histograms
    cut_pixel_threshold: float = 30.0  # mean abs difference of 64x36 grayscale thumbnails (0-255)
    lock_threshold: float = 0.05  # if the face barely moves within a shot, lock the camera
    deadzone: float = 0.04  # ignore target moves smaller than this (fraction of frame width)
    max_speed: float = 0.35  # max pan speed, frame widths per second
    smooth_seconds: float = 0.5
    two_shot: bool = True  # if two faces fit into the crop, frame both


@dataclass
class CaptionsCfg:
    enabled: bool = True
    font: str = "Arial Black"
    font_size: int = 84
    uppercase: bool = True
    max_words: int = 3
    max_chars: int = 18
    color: str = "FFFFFF"  # RGB hex
    highlight_color: str = "FFE600"
    outline: int = 6
    shadow: int = 2
    position: float = 0.70  # vertical position of the text baseline, fraction of height


@dataclass
class RenderCfg:
    encoder: str = "h264_nvenc"  # falls back to libx264 if NVENC is unavailable
    cq: int = 21
    preset: str = "p5"
    audio_bitrate: str = "192k"
    pad_start: float = 0.15
    pad_end: float = 0.4


@dataclass
class Config:
    workdir: str = "work"
    whisper: WhisperCfg = field(default_factory=WhisperCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    highlights: HighlightsCfg = field(default_factory=HighlightsCfg)
    reframe: ReframeCfg = field(default_factory=ReframeCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    render: RenderCfg = field(default_factory=RenderCfg)


def _merge(obj: Any, data: dict, path: str = "") -> None:
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {path}{key} must be a mapping")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: Optional[Path]) -> Config:
    cfg = Config()
    if path is None:
        default = Path("config.yaml")
        path = default if default.exists() else None
    if path is not None:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    return cfg


def save_config(cfg: Config, path: Path) -> None:
    import yaml

    data = dataclasses.asdict(cfg)
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)
