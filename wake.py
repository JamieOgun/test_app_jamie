"""Wake on the robot's name: local speech detection, remote transcription, name match.

The same approach as the devkit's wakeword example
(palmimo-devkit/examples/agents/wakeword): Silero VAD on the robot cuts the mic
audio into utterances, each utterance is transcribed by OpenAI, and
``palmimo_sdk.NameMatcher`` checks the transcript for "Palmimo" by sound rather
than spelling, so "Parmimo", "Parumimo" and パルミーモ all count.

Only utterances (not silence) leave the robot, and nothing reaches ElevenLabs
until the name is heard. Everything said near the robot does go to OpenAI,
though; a fully on-device detector can replace :class:`NameWakeDetector`
behind the :class:`WakeDetector` interface later.

``SileroVad`` and ``UtteranceSegmenter`` are copied from the devkit example's
vad.py (Apache-2.0, Copyright 2026 Jizai Inc.), trimmed of test seams.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

import numpy as np
from palmimo_sdk import NameMatcher
from palmimo_sdk.audio import default_model_dir
from palmimo_sdk.download import download_atomic

logger = logging.getLogger("wake")

SAMPLE_RATE = 16000
#: Frame size the Silero VAD v5 model expects (32 ms at 16 kHz).
FRAME_SAMPLES = 512
SILERO_VAD_MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
SILERO_VAD_MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
#: Shorter utterances (a cough, a door) are not worth a transcription call;
#: "Palmimo" alone takes about 0.6 s.
MIN_UTTERANCE_SECONDS = 0.4


class SileroVad:
    """Speech probability for one 512-sample frame, from the Silero VAD v5 ONNX model."""

    def __init__(self) -> None:
        import onnxruntime as ort

        path = default_model_dir() / "silero_vad.onnx"
        if not path.exists():
            download_atomic(SILERO_VAD_MODEL_URL, path, timeout=60.0, sha256=SILERO_VAD_MODEL_SHA256)
        # One thread and no spinning: the model runs every 32 ms, and the
        # defaults made it use 242% of a Pi 5 core instead of 2.5% (measured
        # in the devkit example).
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(str(path), sess_options=options)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        inputs = {
            "input": (frame.astype(np.float32) / 32768.0)[None, :],
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        output, self._state = self._session.run(None, inputs)
        return float(output[0][0])


class UtteranceSegmenter:
    """Turns a stream of int16 chunks into complete utterances.

    While idle, the last ``pre_roll_seconds`` are kept so the start of speech
    is not clipped. An utterance starts when the speech probability reaches
    ``start_threshold`` and ends after ``silence_seconds`` below
    ``end_threshold``, or at ``max_utterance_seconds``.
    """

    def __init__(
        self,
        vad: SileroVad,
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        silence_seconds: float = 0.8,
        max_utterance_seconds: float = 15.0,
        pre_roll_seconds: float = 0.3,
    ) -> None:
        self._vad = vad
        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._silence_limit = int(silence_seconds * SAMPLE_RATE)
        self._max_samples = int(max_utterance_seconds * SAMPLE_RATE)
        self._pre_roll_limit = int(pre_roll_seconds * SAMPLE_RATE)
        self._reset_state()

    def _reset_state(self) -> None:
        self._carry = np.empty(0, dtype=np.int16)
        self._pre_roll: collections.deque[np.ndarray] = collections.deque()
        self._pre_roll_total = 0
        self._speech: list[np.ndarray] = []
        self._speech_total = 0
        self._speaking = False
        self._silence_run = 0

    def reset(self) -> None:
        """Drop everything buffered, including the VAD's recurrent state."""
        self._reset_state()
        self._vad.reset()

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """Consume *chunk*; return a completed utterance, or None."""
        buffer = np.concatenate([self._carry, chunk]) if len(self._carry) else chunk
        n_frames = len(buffer) // FRAME_SAMPLES
        self._carry = buffer[n_frames * FRAME_SAMPLES :].copy().astype(np.int16)
        for i in range(n_frames):
            result = self._feed_frame(buffer[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
            if result is not None:
                # Keep the rest of this chunk: it may open the next utterance.
                self._carry = buffer[(i + 1) * FRAME_SAMPLES :].copy().astype(np.int16)
                return result
        return None

    def _feed_frame(self, frame: np.ndarray) -> np.ndarray | None:
        prob = self._vad(frame)
        if not self._speaking:
            if prob < self._start_threshold:
                self._pre_roll.append(frame)
                self._pre_roll_total += len(frame)
                while self._pre_roll_total > self._pre_roll_limit and self._pre_roll:
                    self._pre_roll_total -= len(self._pre_roll.popleft())
                return None
            self._speaking = True
            self._speech = [*self._pre_roll, frame]
            self._speech_total = self._pre_roll_total + len(frame)
            self._pre_roll.clear()
            self._pre_roll_total = 0
            self._silence_run = 0
        else:
            self._speech.append(frame)
            self._speech_total += len(frame)
            self._silence_run = 0 if prob >= self._end_threshold else self._silence_run + len(frame)
        if self._silence_run >= self._silence_limit or self._speech_total >= self._max_samples:
            utterance = np.concatenate(self._speech)
            self._speech, self._speech_total, self._speaking, self._silence_run = [], 0, False, 0
            self._vad.reset()
            return utterance
        return None


def encode_wav(samples: np.ndarray) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


@dataclass
class WakeEvent:
    """The robot was called. *command* is what was said after the name, possibly empty."""

    command: str
    transcript: str


class WakeDetector(Protocol):
    """Anything that turns 16 kHz mono int16 audio into wake events."""

    async def feed(self, chunk: np.ndarray) -> WakeEvent | None: ...

    def reset(self) -> None: ...


class NameWakeDetector:
    """Wakes when an utterance contains the robot's name.

    Transcription runs in a thread, so the event loop keeps reading the mic
    meanwhile. The language is not fixed: Japanese and English both work.
    """

    def __init__(self, client: Any, stt_model: str) -> None:
        self._client = client  # an openai.OpenAI
        self._stt_model = stt_model
        self._segmenter = UtteranceSegmenter(SileroVad())
        self._matcher = NameMatcher()

    def reset(self) -> None:
        self._segmenter.reset()

    async def feed(self, chunk: np.ndarray) -> WakeEvent | None:
        utterance = self._segmenter.feed(chunk)
        if utterance is None or len(utterance) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
            return None
        try:
            text = await asyncio.to_thread(self._transcribe, utterance)
        except Exception as exc:  # noqa: BLE001 --  # network, quota, bad key: keep listening
            logger.warning("transcription failed: %s", exc)
            return None
        match = self._matcher.match(text)
        if match is None:
            logger.debug("heard (not for me): %s", text)
            return None
        logger.info("woken by %r -> command %r", text, match.command)
        return WakeEvent(command=match.command, transcript=text)

    def _transcribe(self, utterance: np.ndarray) -> str:
        file = BytesIO(encode_wav(utterance))
        file.name = "utterance.wav"
        return self._client.audio.transcriptions.create(model=self._stt_model, file=file).text.strip()
