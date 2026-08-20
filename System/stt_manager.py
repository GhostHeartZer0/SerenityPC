# System/stt_manager.py
# Offline-first Speech-to-Text Manager using sounddevice and local audio transcription.

import os
import io
import wave
import threading
import time
from typing import Optional, Callable, Dict, Any, List

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class STTManager:
    """
    Manages audio capture via sounddevice and local transcription with 100% offline compliance.
    """
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self._audio_chunks: List[np.ndarray] = []
        self._record_stream = None
        self._record_thread: Optional[threading.Thread] = None
        self._recognizer = sr.Recognizer() if sr is not None else None

    @staticmethod
    def is_available() -> bool:
        """Returns True if required audio dependencies are present."""
        return sd is not None and np is not None and sr is not None

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        """Returns list of available host audio input devices."""
        if sd is None:
            return []
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "id": idx,
                        "name": dev.get("name", f"Input Device {idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": int(dev.get("default_samplerate", 16000))
                    })
        except Exception as e:
            print(f"[STT] Failed to query audio input devices: {e}")
        return devices

    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Starts background microphone audio capture."""
        if not self.is_available() or self.is_recording:
            return False

        try:
            self._audio_chunks = []
            self.is_recording = True

            def _audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"[STT] Audio Stream Status: {status}")
                if self.is_recording:
                    self._audio_chunks.append(indata.copy())

            self._record_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=device_index,
                callback=_audio_callback
            )
            self._record_stream.start()
            print("[STT] Voice recording started.")
            return True
        except Exception as e:
            print(f"[STT] Failed to start audio recording: {e}")
            self.is_recording = False
            if self._record_stream:
                try:
                    self._record_stream.close()
                except Exception:
                    pass
                self._record_stream = None
            return False

    def stop_recording(self) -> Optional[bytes]:
        """
        Stops recording and returns raw WAV byte stream.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        if self._record_stream:
            try:
                self._record_stream.stop()
                self._record_stream.close()
            except Exception as e:
                print(f"[STT] Error closing record stream: {e}")
            self._record_stream = None

        if not self._audio_chunks:
            print("[STT] No audio data recorded.")
            return None

        try:
            audio_data = np.concatenate(self._audio_chunks, axis=0)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_bytes = wav_io.getvalue()
            print(f"[STT] Recording captured {len(wav_bytes)} bytes WAV PCM.")
            return wav_bytes
        except Exception as e:
            print(f"[STT] Error packaging audio to WAV: {e}")
            return None

    def transcribe_wav_bytes(
        self,
        wav_bytes: bytes,
        language: str = "en-US",
        on_complete: Optional[Callable[[str, Optional[str]], None]] = None,
        llm_model: Optional[Any] = None
    ) -> None:
        """
        Transcribes WAV bytes completely offline on a background thread.
        If local Sphinx/Vosk is unavailable, falls back to multimodal ASR if an audio-capable model is loaded.
        """
        def _worker():
            transcript = ""
            error_msg = None
            try:
                if not wav_bytes or sr is None or self._recognizer is None:
                    raise RuntimeError("Speech recognition subsystem unavailable.")

                wav_io = io.BytesIO(wav_bytes)
                with sr.AudioFile(wav_io) as source:
                    audio = self._recognizer.record(source)

                # 1. Attempt offline Sphinx if installed
                try:
                    transcript = self._recognizer.recognize_sphinx(audio, language=language)
                    print(f"[STT] Offline Sphinx transcript: {transcript}")
                except (sr.RequestError, AttributeError, Exception) as sphinx_err:
                    # 2. Check for local Vosk offline model
                    try:
                        transcript = self._recognizer.recognize_vosk(audio)
                        import json
                        vosk_dict = json.loads(transcript) if transcript.startswith("{") else {}
                        transcript = vosk_dict.get("text", transcript)
                        print(f"[STT] Offline Vosk transcript: {transcript}")
                    except Exception:
                        # 3. Multimodal Audio ASR via loaded LLM model if available
                        if llm_model and hasattr(llm_model, "create_chat_completion"):
                            try:
                                import base64
                                b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
                                msgs = [
                                    {"role": "user", "content": [
                                        {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}},
                                        {"type": "text", "text": "Transcribe the following speech segment into text. Output only the verbatim transcription."}
                                    ]}
                                ]
                                res = llm_model.create_chat_completion(messages=msgs, max_tokens=256, temperature=0.1)
                                transcript = res["choices"][0]["message"]["content"].strip()
                                print(f"[STT] Local Multimodal LLM ASR transcript: {transcript}")
                            except Exception as llm_err:
                                error_msg = f"Offline transcription requires CMU Sphinx, Vosk, or a loaded multimodal model. ({llm_err})"
                        else:
                            error_msg = "Offline STT active. Install pocketsphinx / vosk or load an audio-capable model for local ASR."

            except Exception as e:
                error_msg = str(e)

            if on_complete:
                on_complete(transcript.strip(), error_msg)

        threading.Thread(target=_worker, daemon=True).start()
