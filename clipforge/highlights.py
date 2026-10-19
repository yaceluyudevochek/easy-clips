"""Find viral moments in a transcript with an LLM."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .config import HighlightsCfg
from .llm import LLMClient
from .progress import Reporter

log = logging.getLogger(__name__)

SCORE_KEYS = ("hook_score", "payoff_score", "standalone_score", "emotion_score", "viral_score")
WEIGHTS = {"viral_score": 0.4, "hook_score": 0.2, "payoff_score": 0.15, "standalone_score": 0.15, "emotion_score": 0.1}

FIND_SYSTEM = """You are a top short-form video editor (TikTok, Reels, YouTube Shorts).
You receive a fragment of a long video's transcript, split into numbered sentences.
Your job: find the moments that would work as standalone viral short clips.

What makes a great clip:
- HOOK: the very first sentence grabs attention in 1-3 seconds — a bold claim, a question, conflict,
  a surprising fact, a strong emotion, the start of a story. Never start with filler
  ("so", "and", "well", "ну", "вот", "короче") or in the middle of a thought.
- PAYOFF: it ends on a punchline, conclusion, reveal or strong statement. Never end mid-thought.
- STANDALONE: a viewer who has not seen the rest of the video understands it.
- EMOTION / VALUE: humor, controversy, a personal story, a counter-intuitive insight, a practical tip.
Sentences marked with "!" are noticeably louder than average (laughter, excitement, raised voice).
That is a hint, not a rule.

Rules:
- A clip is a contiguous range of sentences: start_id..end_id (inclusive).
- Respect the duration limits given by the user; use the timestamps to estimate duration.
- Quality over quantity: return at most {per_chunk} clips, or an empty list if nothing is good.
- Score each criterion (*_score fields) from 0 to 10 honestly; most moments in a video are NOT viral (score 3-6).
- Write "title" (a catchy short caption, max 8 words) and "hook" (why the first seconds work)
  in the same language as the transcript."""

FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_id": {"type": "integer"},
                    "end_id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    **{k: {"type": "integer"} for k in SCORE_KEYS},
                },
                "required": ["start_id", "end_id", "reason", "hook", "title", *SCORE_KEYS],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

RERANK_SYSTEM = """You are a top short-form video editor. You get candidate clips cut from one long video.
Compare them against each other and rate each one's potential to go viral as a standalone short
(TikTok / Reels / Shorts) on a 0-100 scale. Be strict and spread the scores out: the best clip should
clearly stand out, weak or confusing clips should get low scores. Judge the hook in the first seconds,
whether it is understandable without context, emotional impact and the ending."""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["index", "reason", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


def format_sentences(sents: List[dict]) -> str:
    lines = []
    for s in sents:
        mark = " !" if s.get("energy", 0.0) >= 1.0 else ""
        lines.append(f"[{s['id']}] {s['s']:.1f}s{mark} {s['text']}")
    return "\n".join(lines)


def chunk_sentences(sents: List[dict], chunk_sec: float, overlap_sec: float) -> List[List[dict]]:
    chunks: List[List[dict]] = []
    i = 0
    while i < len(sents):
        t0 = sents[i]["s"]
        j = i
        while j < len(sents) and sents[j]["e"] - t0 <= chunk_sec:
            j += 1
        j = max(j, i + 1)
        chunks.append(sents[i:j])
        if j >= len(sents):
            break
        t_next = sents[j - 1]["e"] - overlap_sec
        k = i + 1
        while k < j and sents[k]["s"] < t_next:
            k += 1
        i = k
    return chunks


def fit_duration(a: int, b: int, sents: List[dict], cfg: HighlightsCfg) -> Optional[Tuple[int, int]]:
    """Adjust the sentence range to the duration limits; trims/extends the end to keep the hook."""
    n = len(sents)
    a, b = max(0, min(a, n - 1)), max(0, min(b, n - 1))
    if b < a:
        a, b = b, a

    def dur(x: int, y: int) -> float:
        return sents[y]["e"] - sents[x]["s"]

    while dur(a, b) > cfg.max_duration and b > a:
        b -= 1
    if dur(a, b) > cfg.max_duration:
        return None
    while dur(a, b) < cfg.min_duration and b + 1 < n and dur(a, b + 1) <= cfg.max_duration:
        b += 1
    if dur(a, b) < cfg.min_duration:
        return None
    return a, b


def composite(scores: dict) -> float:
    return round(sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS), 2)


def overlap_ratio(x: dict, y: dict) -> float:
    inter = min(x["end"], y["end"]) - max(x["start"], y["start"])
    if inter <= 0:
        return 0.0
    return inter / min(x["end"] - x["start"], y["end"] - y["start"])


def dedupe(cands: List[dict], max_overlap: float) -> List[dict]:
    kept: List[dict] = []
    for c in sorted(cands, key=lambda c: c["score"], reverse=True):
        if all(overlap_ratio(c, k) <= max_overlap for k in kept):
            kept.append(c)
    return kept


def _make_candidate(raw: dict, sents: List[dict], cfg: HighlightsCfg) -> Optional[dict]:
    try:
        fitted = fit_duration(int(raw["start_id"]), int(raw["end_id"]), sents, cfg)
    except (KeyError, TypeError, ValueError):
        return None
    if fitted is None:
        return None
    a, b = fitted
    scores = {k: max(0, min(10, int(raw.get(k, 0) or 0))) for k in SCORE_KEYS}
    return {
        "sent_a": a, "sent_b": b,
        "start": sents[a]["s"], "end": sents[b]["e"],
        "title": str(raw.get("title", "")).strip(),
        "hook": str(raw.get("hook", "")).strip(),
        "reason": str(raw.get("reason", "")).strip(),
        "scores": scores,
        "score": composite(scores),
        "text": " ".join(s["text"] for s in sents[a:b + 1]),
    }


def find_candidates(sents: List[dict], client: LLMClient, cfg: HighlightsCfg,
                    report: Reporter = Reporter()) -> List[dict]:
    chunks = chunk_sentences(sents, cfg.chunk_minutes * 60, cfg.chunk_overlap_minutes * 60)
    system = FIND_SYSTEM.replace("{per_chunk}", str(cfg.per_chunk))
    cands: List[dict] = []
    for i, chunk in enumerate(chunks, 1):
        report("highlights", i - 1, len(chunks))
        log.info("Searching moments: chunk %d/%d (%.0f-%.0f s)", i, len(chunks), chunk[0]["s"], chunk[-1]["e"])
        user = (
            f"Clip duration limits: {cfg.min_duration:.0f}-{cfg.max_duration:.0f} seconds.\n"
            f"Transcript fragment, format: [sentence_id] start_time text\n\n{format_sentences(chunk)}"
        )
        try:
            result = client.chat_json(system, user, FIND_SCHEMA)
        except Exception as e:
            log.warning("Chunk %d skipped: %s", i, e)
            continue
        for raw in result.get("clips", []):
            c = _make_candidate(raw, sents, cfg)
            if c:
                cands.append(c)
    log.info("LLM proposed %d valid candidates", len(cands))
    return cands


def rerank(cands: List[dict], client: LLMClient) -> List[dict]:
    blocks = []
    for i, c in enumerate(cands):
        blocks.append(f"### Candidate {i} ({c['end'] - c['start']:.0f} s)\nTitle: {c['title']}\n{c['text']}")
    user = "Rate every candidate below.\n\n" + "\n\n".join(blocks)
    try:
        result = client.chat_json(RERANK_SYSTEM, user, RERANK_SCHEMA)
    except Exception as e:
        log.warning("Rerank failed, keeping first-pass scores: %s", e)
        return cands
    for item in result.get("ranking", []):
        try:
            idx, score = int(item["index"]), int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(cands):
            c = cands[idx]
            c["rerank_score"] = max(0, min(100, score))
            c["rerank_reason"] = str(item.get("reason", ""))
            # Blend: comparative judgement dominates, first-pass score breaks ties.
            c["score"] = round(0.7 * c["rerank_score"] / 10 + 0.3 * c["score"], 2)
    return sorted(cands, key=lambda c: c["score"], reverse=True)


def pad_bounds(clip: dict, sents: List[dict], words: List[dict], pad_start: float, pad_end: float) -> None:
    """Add breathing room around the clip without bleeding into neighbouring words."""
    w0, w1 = sents[clip["sent_a"]]["w0"], sents[clip["sent_b"]]["w1"]
    start = clip["start"] - pad_start
    if w0 > 0:
        start = max(start, words[w0 - 1]["e"])
    end = clip["end"] + pad_end
    if w1 + 1 < len(words):
        end = min(end, words[w1 + 1]["s"])
    clip["start"], clip["end"] = round(max(0.0, start), 3), round(end, 3)


def select_highlights(sents: List[dict], words: List[dict], client: LLMClient, cfg: HighlightsCfg,
                      pad_start: float, pad_end: float, max_clips: Optional[int] = None,
                      report: Reporter = Reporter()) -> List[dict]:
    max_clips = max_clips or cfg.max_clips
    cands = dedupe(find_candidates(sents, client, cfg, report), cfg.max_overlap)
    cands = [c for c in cands if c["score"] >= cfg.min_score] or cands[:max_clips]
    if cfg.rerank and len(cands) > 1:
        log.info("Reranking top %d candidates", min(len(cands), cfg.rerank_top))
        report("rerank")
        cands = rerank(cands[: cfg.rerank_top], client)
    clips = cands[:max_clips]
    for c in clips:
        pad_bounds(c, sents, words, pad_start, pad_end)
    return clips
