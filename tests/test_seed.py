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

    def __init__(self):
        self.caches = []  # prompt caches it was given, in order
        self.references = []  # (audio, prompt_text) of the voices it encoded

    def build_prompt_cache(self, reference_wav_path, prompt_text=None, prompt_wav_path=None):
        import soundfile as sf
        self.references.append((sf.read(reference_wav_path, dtype="float32")[0], prompt_text))
        return {"mode": "ref_continuation"}

    def generate_with_prompt_cache(self, target_text, prompt_cache, cfg_value, inference_timesteps):
        self.caches.append(prompt_cache)
        feat = torch.randn(4, 8)
        return torch.randn(1, 1600), None, feat

    def generate_with_prompt_cache_streaming(self, target_text, prompt_cache, cfg_value, inference_timesteps):
        """Like voxcpm: one decoded patch per step, plus the growing list of [1, 1, p, d] features."""
        self.caches.append(prompt_cache)
        feats = [torch.randn(1, 1, 2, 4) for _ in range(2)]  # prompt context patches
        for _ in range(3):
            feats.append(torch.randn(1, 1, 2, 4))
            yield torch.randn(1, 160), None, feats


class FakeVoxCPM:
    tts_model = FakeTTSModel()

    def generate(self, text, **kw):
        return torch.randn(1600).numpy()

    def generate_streaming(self, text, **kw):
        for _ in range(3):
            yield torch.randn(800).numpy()


class FakeMgr:
    def __init__(self):
        self.voxcpm = FakeVoxCPM()
        self.voxcpm.tts_model = FakeTTSModel()
    sample_rate = 16000


class ServerTest(unittest.TestCase):
    def setUp(self):
        self._orig = server.get_mgr
        self.mgr = FakeMgr()
        server.get_mgr = lambda: self.mgr
        server._load_prompt_cache = lambda voice_id: {"mode": "continuation"}
        self.client = TestClient(server.app)

    def tearDown(self):
        server.get_mgr = self._orig

    def post(self, path, **data):
        r = self.client.post(path, data=data)
        self.assertEqual(r.status_code, 200, r.text)
        return r



class SeedTest(ServerTest):
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


class CloneStreamingTest(ServerTest):
    """/api/voice-clone/stream streams each sentence patch by patch (first audio after ~0.2 s, not
    after the whole sentence); the next sentence continues from the whole previous one."""

    def test_clone_audio_arrives_in_patches(self):
        r = self.post("/api/voice-clone/stream", target_text="你好。", voice_id="v", seed="1")
        self.assertEqual(len(r.content) - 44, 3 * 160 * 2)  # three 160-sample patches, int16

    def test_next_sentence_continues_from_the_whole_previous_one(self):
        long = "第一句" + "很长" * 30 + "。第二句" + "也很长" * 20 + "。"  # over CHUNK_MAX_CHARS: two chunks
        self.post("/api/voice-clone/stream", target_text=long, voice_id="v")
        caches = self.mgr.voxcpm.tts_model.caches
        self.assertEqual(len(caches), 2)
        self.assertEqual(tuple(caches[1]["audio_feat"].shape), (5, 2, 4))  # context + 3 new patches

    def test_speed_change_needs_the_whole_sentence(self):
        r = self.post("/api/voice-clone/stream", target_text="你好。", voice_id="v", speed="1.3")
        self.assertNotEqual(len(r.content) - 44, 3 * 160 * 2)


class DesignedVoiceTest(ServerTest):
    """Voice design draws a new speaker for every sentence, even with one seed (the seed only
    repeats the same sentence): /api/voices/from-design saves one designed sentence as a voice."""

    def setUp(self):
        super().setUp()
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self._voices = server.VOICES_DIR
        server.VOICES_DIR = Path(self.tmp.name)

    def tearDown(self):
        server.VOICES_DIR = self._voices
        self.tmp.cleanup()
        super().tearDown()

    def test_the_saved_voice_is_the_sentence_the_stream_spoke(self):
        heard = self.post("/api/voice-design/stream", text="大家好", description="年轻女性", seed="42").content
        meta = self.post("/api/voices/from-design", text="大家好", description="年轻女性", seed="42").json()
        audio, text = self.mgr.voxcpm.tts_model.references[-1]
        pcm = np.frombuffer(heard[44:], np.int16) / 32768
        self.assertEqual(len(audio), len(pcm))
        self.assertLessEqual(np.abs(audio - pcm).max(), 2 / 32768)  # two int16 roundings only
        self.assertEqual((text, meta["has_text"], meta["seed"], meta["description"]), ("大家好", True, 42, "年轻女性"))

    def test_it_is_listed_with_its_name(self):
        meta = self.post("/api/voices/from-design", text="大家好", seed="7", name="主播A").json()
        self.assertEqual([v["name"] for v in self.client.get("/api/voices").json()], ["主播A"])
        self.assertTrue((server.VOICES_DIR / meta["id"] / "meta.json").exists())

    def test_a_seed_is_required(self):
        r = self.client.post("/api/voices/from-design", data={"text": "大家好", "seed": "-1"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
