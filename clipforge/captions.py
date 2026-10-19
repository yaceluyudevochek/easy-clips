"""Word-by-word animated captions as an ASS subtitle file."""
from __future__ import annotations

from typing import List

from .config import CaptionsCfg
from .sentences import _ends_sentence


def _ass_color(rgb: str) -> str:
    rgb = rgb.lstrip("#")
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str, upper: bool) -> str:
    w = word.strip().replace("\\", "").replace("{", "(").replace("}", ")")
    w = w.rstrip(".,;:")
    return w.upper() if upper else w


def clip_words(words: List[dict], start: float, end: float) -> List[dict]:
    """Words inside [start, end], with times relative to the clip start."""
    out = []
    for w in words:
        if w["s"] >= start - 0.05 and w["e"] <= end + 0.1:
            out.append({"w": w["w"], "s": max(0.0, w["s"] - start), "e": min(end, w["e"]) - start})
    return out


def group_words(words: List[dict], cfg: CaptionsCfg) -> List[List[dict]]:
    groups: List[List[dict]] = []
    cur: List[dict] = []
    for w in words:
        text = _clean(w["w"], cfg.uppercase)
        if not text:
            continue
        item = {**w, "text": text}
        if cur:
            chars = sum(len(x["text"]) + 1 for x in cur) + len(text)
            if (len(cur) >= cfg.max_words or chars > cfg.max_chars or w["s"] - cur[-1]["e"] > 0.6
                    or _ends_sentence(cur[-1]["w"])):
                groups.append(cur)
                cur = []
        cur.append(item)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: List[dict], cfg: CaptionsCfg, width: int, height: int) -> str:
    margin_v = int(height * (1 - cfg.position))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg.font},{cfg.font_size},{_ass_color(cfg.color)},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{cfg.outline},{cfg.shadow},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    hl = _ass_color(cfg.highlight_color)
    lines = []
    groups = group_words(words, cfg)
    for gi, group in enumerate(groups):
        g_end = group[-1]["e"] + 0.5
        if gi + 1 < len(groups):
            g_end = min(g_end, groups[gi + 1][0]["s"])
        g_end = max(g_end, group[-1]["e"])
        for k, w in enumerate(group):
            t0 = w["s"] if k else group[0]["s"]
            t1 = group[k + 1]["s"] if k + 1 < len(group) else g_end
            if t1 - t0 < 0.02:
                continue
            parts = [
                f"{{\\c{hl}&\\fscx108\\fscy108}}{x['text']}{{\\r}}" if j == k else x["text"]
                for j, x in enumerate(group)
            ]
            lines.append(f"Dialogue: 0,{_ts(t0)},{_ts(t1)},Default,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"
