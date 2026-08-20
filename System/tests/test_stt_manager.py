# System/tests/test_stt_manager.py
import unittest
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
import io
import wave
from System.stt_manager import STTManager

class TestSTTManager(unittest.TestCase):
    def setUp(self):
        self.stt = STTManager(sample_rate=16000, channels=1)

    def test_availability(self):
        self.assertTrue(STTManager.is_available())

    def test_get_input_devices(self):
        devices = STTManager.get_input_devices()
        self.assertIsInstance(devices, list)
        if devices:
            self.assertIn("id", devices[0])
            self.assertIn("name", devices[0])

    def test_wav_synthesis_and_parsing(self):
        # Generate 0.5s of synthetic sine wave PCM audio
        duration = 0.5
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(440 * 2 * np.pi * t)
        audio_int16 = (tone * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        wav_bytes = wav_io.getvalue()
        self.assertTrue(len(wav_bytes) > 0)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

        # Verify transcription worker gracefully handles offline transcribe
        result_holder = []
        def _cb(text, err):
            result_holder.append((text, err))

        self.stt.transcribe_wav_bytes(wav_bytes, language="en-US", on_complete=_cb)
        import time
        for _ in range(20):
            if result_holder:
                break
            time.sleep(0.1)

        self.assertTrue(len(result_holder) > 0)

if __name__ == "__main__":
    unittest.main()
