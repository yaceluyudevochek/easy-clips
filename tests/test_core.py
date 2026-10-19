"""Unit tests for the pure-logic parts: python -m unittest discover tests"""
import unittest

import numpy as np

from clipforge.captions import build_ass, group_words
from clipforge.config import CaptionsCfg, HighlightsCfg, ReframeCfg
from clipforge.highlights import chunk_sentences, dedupe, fit_duration, select_highlights
from clipforge.llm import parse_json
from clipforge.media import VideoInfo
from clipforge.reframe import plan_camera
from clipforge.render import crop_commands
from clipforge.sentences import build_sentences


def make_words(n_sent=40, words_per=8, word_len=0.4):
    words, t = [], 0.0
    for s in range(n_sent):
        for k in range(words_per):
            w = f" w{s}_{k}" + ("." if k == words_per - 1 else "")
            words.append({"w": w, "s": round(t, 3), "e": round(t + word_len - 0.05, 3), "p": 0.9})
            t += word_len
        t += 0.3
    return words


class FakeClient:
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, schema=None):
        self.calls += 1
        if "Rate every candidate" in user:
            return {"ranking": [{"index": 0, "reason": "ok", "score": 40}, {"index": 1, "reason": "best", "score": 95}]}
        ids = [int(line[1:line.index("]")]) for line in user.splitlines() if line.startswith("[")]
        a = ids[0]
        return {"clips": [
            {"start_id": a, "end_id": a + 7, "reason": "r", "hook": "h", "title": f"T{a}",
             "hook_score": 8, "payoff_score": 7, "standalone_score": 8, "emotion_score": 6, "viral_score": 8},
            {"start_id": a + 1, "end_id": a + 8, "reason": "dup", "hook": "h", "title": "dup",
             "hook_score": 5, "payoff_score": 5, "standalone_score": 5, "emotion_score": 5, "viral_score": 5},
        ]}


class TestSentences(unittest.TestCase):
    def test_split_on_punctuation_and_pause(self):
        words = make_words(3, 4)
        sents = build_sentences(words)
        self.assertEqual(len(sents), 3)
        self.assertEqual(sents[1]["w0"], 4)
        self.assertTrue(sents[0]["text"].endswith("."))
        words[2]["s"] += 2.0  # long pause inside the first sentence
        words[2]["e"] += 2.0
        self.assertEqual(len(build_sentences(words)), 4)


class TestHighlights(unittest.TestCase):
    def setUp(self):
        self.words = make_words()
        self.sents = build_sentences(self.words)
        self.cfg = HighlightsCfg(min_duration=10, max_duration=30, chunk_minutes=2, chunk_overlap_minutes=0.5)

    def test_chunks_cover_everything_with_overlap(self):
        chunks = chunk_sentences(self.sents, 120, 30)
        ids = {s["id"] for c in chunks for s in c}
        self.assertEqual(ids, set(range(len(self.sents))))
        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1][0]["s"], chunks[0][-1]["e"])

    def test_fit_duration(self):
        a, b = fit_duration(0, 30, self.sents, self.cfg)
        self.assertLessEqual(self.sents[b]["e"] - self.sents[a]["s"], 30)
        a, b = fit_duration(5, 5, self.sents, self.cfg)  # too short -> extended
        self.assertGreaterEqual(self.sents[b]["e"] - self.sents[a]["s"], 10)
        self.assertEqual(a, 5)

    def test_dedupe(self):
        c = [{"start": 0, "end": 20, "score": 5}, {"start": 2, "end": 22, "score": 7}, {"start": 30, "end": 50, "score": 6}]
        kept = dedupe(c, 0.3)
        self.assertEqual([k["score"] for k in kept], [7, 6])

    def test_select_end_to_end_with_fake_llm(self):
        client = FakeClient()
        clips = select_highlights(self.sents, self.words, client, self.cfg, 0.15, 0.4, max_clips=3)
        self.assertTrue(1 <= len(clips) <= 3)
        for c in clips:
            self.assertGreater(c["end"], c["start"])
            self.assertEqual(c["hook"], "h")
        self.assertEqual(clips[0]["rerank_score"], 95)


class TestSchema(unittest.TestCase):
    def test_no_key_collisions(self):
        from clipforge.highlights import FIND_SCHEMA

        props = FIND_SCHEMA["properties"]["clips"]["items"]["properties"]
        self.assertEqual(props["hook"]["type"], "string")
        self.assertEqual(props["viral_score"]["type"], "integer")


class TestParseJson(unittest.TestCase):
    def test_think_and_fences(self):
        self.assertEqual(parse_json('<think>hmm {x}</think>\n```json\n{"a": 1}\n```'), {"a": 1})


class TestCaptions(unittest.TestCase):
    def test_groups_and_ass(self):
        words = [{"w": f" word{i}", "s": i * 0.3, "e": i * 0.3 + 0.25} for i in range(7)]
        cfg = CaptionsCfg(max_words=3)
        groups = group_words(words, cfg)
        self.assertEqual([len(g) for g in groups], [3, 3, 1])
        ass = build_ass(words, cfg, 1080, 1920)
        self.assertEqual(ass.count("Dialogue:"), 7)
        self.assertIn("WORD0", ass)
        self.assertIn("PlayResY: 1920", ass)


class TestCamera(unittest.TestCase):
    def test_locks_on_face_and_jumps_on_cut(self):
        fps = 8
        n = 80
        faces = [[(0.3, 0.4, 0.1, 0.18, 0.95)] if i < 40 else [(0.75, 0.4, 0.1, 0.18, 0.95)] for i in range(n)]
        analysis = {"times": [i / fps for i in range(n)], "faces": faces, "cuts": [4.96]}
        keys = plan_camera(analysis, 0.316, ReframeCfg(analysis_fps=fps))
        kt = np.array([k[0] for k in keys])
        kx = np.array([k[1] for k in keys])
        self.assertAlmostEqual(float(np.interp(2.0, kt, kx)), 0.3, places=2)
        self.assertAlmostEqual(float(np.interp(4.95, kt, kx)), 0.3, places=2)
        self.assertAlmostEqual(float(np.interp(4.97, kt, kx)), 0.75, places=2)
        info = VideoInfo(1920, 1080, 30.0, 10.0)
        x0, script = crop_commands(keys, info, 606, 10.0)
        self.assertEqual(x0, int(round(0.3 * 1920 - 303)) // 2 * 2)
        self.assertEqual(len(script.strip().splitlines()), 1)  # one jump, otherwise static

    def test_no_faces_centers(self):
        analysis = {"times": [i / 8 for i in range(16)], "faces": [[] for _ in range(16)], "cuts": []}
        keys = plan_camera(analysis, 0.316, ReframeCfg())
        self.assertTrue(all(abs(x - 0.5) < 1e-9 for _, x in keys))


if __name__ == "__main__":
    unittest.main()
