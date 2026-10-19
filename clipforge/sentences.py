"""Group Whisper words into sentences and annotate them with audio energy."""
from __future__ import annotations

import wave
from pathlib import Path
from typing import List

END_PUNCT = ".!?…。！？"
TRAILING = "\"'»”’)]"


def _ends_sentence(token: str) -> bool:
    t = token.strip().rstrip(TRAILING)
    return bool(t) and t[-1] in END_PUNCT


def build_sentences(words: List[dict], max_pause: float = 0.8, max_words: int = 40, max_dur: float = 15.0) -> List[dict]:
    sentences: List[dict] = []
    start_idx = 0

    def flush(end_idx: int) -> None:
        nonlocal start_idx
        if end_idx < start_idx:
            return
        chunk = words[start_idx:end_idx + 1]
        text = "".join(w["w"] for w in chunk).strip()
        if text:
            sentences.append({
                "id": len(sentences), "s": chunk[0]["s"], "e": chunk[-1]["e"],
                "w0": start_idx, "w1": end_idx, "text": text,
            })
        start_idx = end_idx + 1

    for i, w in enumerate(words):
        if i > start_idx and w["s"] - words[i - 1]["e"] > max_pause:
            flush(i - 1)
        n = i - start_idx + 1
        if _ends_sentence(w["w"]) or n >= max_words or w["e"] - words[start_idx]["s"] >= max_dur:
            flush(i)
    flush(len(words) - 1)
    return sentences


def annotate_energy(sentences: List[dict], wav_path: Path, window: float = 0.25) -> None:
    """Adds sentence['energy'] — loudness z-score relative to the whole video."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sr * window))
    n = len(audio) // hop
    if n == 0:
        return
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    voiced = db[db > np.percentile(db, 20)]  # ignore silence when computing the baseline
    mean, std = float(voiced.mean()), float(voiced.std() or 1.0)
    for s in sentences:
        a, b = int(s["s"] / window), max(int(s["s"] / window) + 1, int(s["e"] / window))
        seg = db[a:b]
        s["energy"] = round((float(np.percentile(seg, 90)) - mean) / std, 2) if len(seg) else 0.0
