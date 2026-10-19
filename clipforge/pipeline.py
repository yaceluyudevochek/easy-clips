"""End-to-end pipeline with per-stage caching in work/<video>/."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Set

from .captions import build_ass, clip_words
from .config import Config
from .highlights import select_highlights
from .llm import LLMClient, LocalServer
from .progress import Reporter
from .media import download, extract_audio, is_url, probe, require_ffmpeg
from .reframe import ensure_face_model, reframe
from .render import render_clip
from .sentences import annotate_energy, build_sentences
from .transcribe import transcribe

log = logging.getLogger(__name__)

STAGES = ("transcribe", "highlights", "render")


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.U).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)[:max_len].strip("_") or "clip"


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def run_pipeline(source: str, cfg: Config, max_clips: Optional[int] = None, force: Set[str] = frozenset(),
                 report: Optional[Reporter] = None) -> Path:
    """Runs all stages and returns the job directory (work/<video>/)."""
    report = report or Reporter()
    require_ffmpeg()
    t_start = time.time()
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        log.info("Downloading %s", source)
        report("download")
        video = download(source, workdir / "_downloads")
    else:
        video = Path(source)
        if not video.exists():
            raise FileNotFoundError(video)
    job = workdir / slugify(video.stem, 60)
    job.mkdir(parents=True, exist_ok=True)
    _save(job / "source.json", {"video": str(video.resolve())})  # lets the GUI re-render a project later
    info = probe(video)
    log.info("Video: %s, %dx%d @ %.2f fps, %.0f s", video.name, info.width, info.height, info.fps, info.duration)

    # 1. Transcription
    transcript_path = job / "transcript.json"
    if "transcribe" in force or not transcript_path.exists():
        wav = job / "audio.wav"
        report("audio")
        extract_audio(video, wav)
        transcript = transcribe(wav, cfg.whisper, report)
        sents = build_sentences(transcript["words"])
        annotate_energy(sents, wav)
        transcript["sentences"] = sents
        _save(transcript_path, transcript)
        force = set(force) | {"highlights", "render"}
    transcript = _load(transcript_path)
    words, sents = transcript["words"], transcript["sentences"]
    log.info("Transcript: %d words, %d sentences", len(words), len(sents))
    if not sents:
        raise RuntimeError("Empty transcript — nothing to cut")

    # 2. Viral moments (LLM)
    clips_path = job / "clips.json"
    if "highlights" in force or not clips_path.exists():
        client = LLMClient(cfg.llm)
        report("llm_start")
        with LocalServer(cfg.llm, job / "llm_server.log"):
            clips = select_highlights(sents, words, client, cfg.highlights,
                                      cfg.render.pad_start, cfg.render.pad_end, max_clips, report)
        if not clips:
            raise RuntimeError("LLM found no usable moments")
        _save(clips_path, clips)
        force = set(force) | {"render"}
    all_clips = _load(clips_path)
    clips = all_clips[:max_clips] if max_clips else all_clips

    # 3. Reframe + captions + render
    out_dir = job / "clips"
    out_dir.mkdir(exist_ok=True)
    model = ensure_face_model(workdir / "_models")
    for i, clip in enumerate(clips, 1):
        report("render", i - 1, len(clips))
        name = f"{i:02d}_{slugify(clip['title'])}"
        out = out_dir / f"{name}.mp4"
        clip["file"] = f"clips/{out.name}"
        if out.exists() and "render" not in force:
            log.info("[%d/%d] exists: %s", i, len(clips), out.name)
            continue
        log.info("[%d/%d] %.1f-%.1f s (score %.1f): %s", i, len(clips), clip["start"], clip["end"],
                 clip["score"], clip["title"])
        keys = reframe(video, clip["start"], clip["end"], info, cfg.reframe, model)
        ass = None
        if cfg.captions.enabled:
            ass = build_ass(clip_words(words, clip["start"], clip["end"]), cfg.captions,
                            cfg.reframe.output_width, cfg.reframe.output_height)
        tmp = out.with_name(out.stem + ".part.mp4")  # an interrupted render never looks finished
        render_clip(video, info, clip["start"], clip["end"], keys, ass, tmp, cfg.reframe, cfg.render)
        tmp.replace(out)

    _save(clips_path, all_clips)
    _write_report(job, video, clips)
    report("done", 1, 1)
    log.info("Done in %.0f s -> %s", time.time() - t_start, out_dir)
    return job


def _write_report(job: Path, video: Path, clips: list) -> None:
    lines = [f"# Clips from {video.name}\n"]
    for i, c in enumerate(clips, 1):
        s = c.get("scores", {})
        lines.append(
            f"## {i}. {c['title']}  (score {c['score']})\n"
            f"- Time: {c['start']:.1f}-{c['end']:.1f} s ({c['end'] - c['start']:.0f} s)\n"
            f"- Scores: " + ", ".join(f"{k} {v}" for k, v in s.items()) + "\n"
            f"- Hook: {c.get('hook', '')}\n"
            f"- Why: {c.get('rerank_reason') or c.get('reason', '')}\n\n"
            f"> {c['text']}\n"
        )
    (job / "report.md").write_text("\n".join(lines), encoding="utf-8")
