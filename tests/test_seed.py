"""Seed control: the same seed gives the same audio, and every response says which seed it used.

Run: venv/bin/python -m unittest discover tests
The model is replaced by a fake whose output is torch noise, so no GPU or weights are needed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402


class FakeTTSModel:
    sample_rate = 16000

    def generate_with_prompt_cache(self, target_text, prompt_cache, cfg_value, inference_timesteps):
        feat = torch.randn(4, 8)
        return torch.randn(1, 1600), None, feat


class FakeVoxCPM:
    tts_model = FakeTTSModel()

    def generate(self, text, **kw):
        return torch.randn(1600).numpy()

    def generate_streaming(self, text, **kw):
        for _ in range(3):
            yield torch.randn(800).numpy()


class FakeMgr:
    voxcpm = FakeVoxCPM()
    sample_rate = 16000


class SeedTest(unittest.TestCase):
    def setUp(self):
        self._orig = server.get_mgr
        server.get_mgr = lambda: FakeMgr()
        server._load_prompt_cache = lambda voice_id: {"mode": "continuation"}
        self.client = TestClient(server.app)

    def tearDown(self):
        server.get_mgr = self._orig

    def post(self, path, **data):
        r = self.client.post(path, data=data)
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def test_same_seed_same_audio_design_stream(self):
        a = self.post("/api/voice-design/stream", text="你好", seed="42")
        torch.randn(100)  # disturb the global RNG in between
        b = self.post("/api/voice-design/stream", text="你好", seed="42")
        self.assertEqual(a.content, b.content)
        self.assertEqual(a.headers["X-Seed"], "42")

    def test_same_seed_same_audio_clone_stream(self):
        a = self.post("/api/voice-clone/stream", target_text="你好", voice_id="v", seed="7")
        b = self.post("/api/voice-clone/stream", target_text="你好", voice_id="v", seed="7")
        self.assertEqual(a.content, b.content)

    def test_same_seed_same_audio_non_streaming(self):
        a = self.post("/api/voice-design", text="你好", seed="3")
        b = self.post("/api/voice-design", text="你好", seed="3")
        self.assertEqual(a.content, b.content)
        c = self.post("/api/voice-clone", target_text="你好", voice_id="v", seed="3")
        d = self.post("/api/voice-clone", target_text="你好", voice_id="v", seed="3")
        self.assertEqual(c.content, d.content)

    def test_no_seed_picks_one_and_reports_it(self):
        a = self.post("/api/voice-design/stream", text="你好")
        seed = int(a.headers["X-Seed"])
        self.assertGreaterEqual(seed, 0)
        b = self.post("/api/voice-design/stream", text="你好", seed=str(seed))
        self.assertEqual(a.content, b.content)

    def test_different_seeds_differ(self):
        a = self.post("/api/voice-design/stream", text="你好", seed="1")
        b = self.post("/api/voice-design/stream", text="你好", seed="2")
        self.assertNotEqual(a.content, b.content)


if __name__ == "__main__":
    unittest.main()
