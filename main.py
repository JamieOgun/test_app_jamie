"""Talk with an ElevenLabs agent through Palmimo's mic and speaker.

The agent (its voice, prompt, language and LLM) is set up in the ElevenLabs
dashboard; this app only carries audio. The mic is streamed to the agent over
its WebSocket, and the agent's voice is streamed to the speaker through aplay.

The mic is muted while the robot is talking. Without a ReSpeaker there is no
echo cancellation, so the agent would otherwise hear its own voice and
interrupt itself. The cost is that you cannot talk over it.

One run is one conversation. When the agent ends the call (or the Portal stops
the app), the app exits, so it never keeps a paid conversation open by itself.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
import websockets
from palmimo_sdk.io.alsa_devices import resolve_alsa_device

API_KEY_ENV = "ELEVENLABS_API_KEY"
SIGNED_URL = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}"

#: Mic chunk length sent to the agent.
CHUNK_SECONDS = 0.05
#: Keep the mic muted this long after the robot's last sample, to cover the
#: speaker's own buffering and room echo.
MUTE_TAIL_SECONDS = 0.4

logger = logging.getLogger("elevenlabs_agent")


def get_signed_url(agent_id: str, api_key: str) -> str:
    """Ask ElevenLabs for a WebSocket URL for *agent_id*; works for private agents too."""
    request = urllib.request.Request(SIGNED_URL.format(agent_id=agent_id), headers={"xi-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)["signed_url"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"ElevenLabs rejected the agent id or key (HTTP {exc.code}): {detail}") from exc


def pcm_rate(audio_format: str) -> int:
    """Sample rate of a ``pcm_<rate>`` format; other formats (such as ulaw) are not handled."""
    if not audio_format.startswith("pcm_"):
        raise RuntimeError(
            f"agent audio format {audio_format!r} is not supported; choose a PCM format in the agent's settings"
        )
    return int(audio_format.removeprefix("pcm_"))


def find_input_device(hint: str | None) -> int | None:
    """Index of the first input device whose name contains *hint*, or None for the default."""
    if not hint:
        return None
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0 and hint.lower() in device["name"].lower():
            logger.info("Selected mic %r for hint %r", device["name"], hint)
            return index
    names = [d["name"] for d in devices if d["max_input_channels"] > 0]
    logger.warning("No mic matches hint %r; using the default. Mics found: %s", hint, names)
    return None


def resample(pcm: bytes, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling of 16-bit mono PCM; good enough for speech recognition."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if from_rate == to_rate or len(samples) == 0:
        return samples.astype("<i2")
    n_out = round(len(samples) * to_rate / from_rate)
    positions = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(positions, np.arange(len(samples)), samples).astype("<i2")


class Player:
    """Streams 16-bit mono PCM to the speaker through one long-running aplay.

    Writes go through a thread, so a full pipe never blocks the WebSocket loop.
    ``busy_until`` estimates when the queued audio finishes playing, which is
    what mutes the mic.
    """

    def __init__(self, device: str | None, rate: int, volume: float) -> None:
        self._argv = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(rate)]
        if device:
            self._argv += ["-D", device]
        self._rate = rate
        self._volume = volume
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._proc = self._start()
        self.busy_until = 0.0
        threading.Thread(target=self._write_loop, daemon=True).start()

    def _start(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(self._argv, stdin=subprocess.PIPE)

    def play(self, pcm: bytes) -> None:
        if self._volume != 1.0:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * self._volume
            pcm = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
        self.busy_until = max(time.monotonic(), self.busy_until) + len(pcm) / 2 / self._rate
        self._queue.put(pcm)

    def interrupt(self) -> None:
        """Drop what is queued and restart aplay, so the agent stops mid-sentence."""
        while not self._queue.empty():
            self._queue.get_nowait()
        with self._lock:
            self._proc.kill()
            self._proc.wait()
            self._proc = self._start()
        self.busy_until = time.monotonic()

    def close(self) -> None:
        self._queue.put(None)
        with self._lock:
            self._proc.kill()

    def _write_loop(self) -> None:
        while (pcm := self._queue.get()) is not None:
            with self._lock:
                stdin = self._proc.stdin
            try:
                stdin.write(pcm)
                stdin.flush()
            except (BrokenPipeError, ValueError):
                pass  # aplay was restarted by interrupt(); this chunk was meant to be dropped


async def converse(args: argparse.Namespace, api_key: str) -> None:
    url = await asyncio.to_thread(get_signed_url, args.agent_id, api_key)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "conversation_initiation_client_data"}))
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == "conversation_initiation_metadata":
                meta = event["conversation_initiation_metadata_event"]
                break
        out_rate = pcm_rate(meta["agent_output_audio_format"])
        in_rate = pcm_rate(meta["user_input_audio_format"])
        logger.info("conversation %s started (voice %d Hz, mic %d Hz)", meta.get("conversation_id"), out_rate, in_rate)

        speaker = resolve_alsa_device(args.speaker_device or None, kind="playback", log=logger)
        player = Player(speaker, out_rate, args.volume)

        # Open the mic at its own rate and resample: cheap USB mics often cannot open at 16 kHz.
        mic_index = find_input_device(args.mic_device or None)
        mic_rate = int(sd.query_devices(mic_index, "input")["default_samplerate"])
        mic_chunks: asyncio.Queue[bytes] = asyncio.Queue()

        def on_mic(data, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
            loop.call_soon_threadsafe(mic_chunks.put_nowait, bytes(data))

        async def send_mic() -> None:
            while True:
                pcm = resample(await mic_chunks.get(), mic_rate, in_rate)
                if time.monotonic() < player.busy_until + MUTE_TAIL_SECONDS:
                    pcm = np.zeros_like(pcm)  # muted while the robot talks
                await ws.send(json.dumps({"user_audio_chunk": base64.b64encode(pcm.tobytes()).decode()}))

        async def receive() -> None:
            async for message in ws:
                event = json.loads(message)
                kind = event.get("type")
                if kind == "audio":
                    player.play(base64.b64decode(event["audio_event"]["audio_base_64"]))
                elif kind == "interruption":
                    player.interrupt()
                elif kind == "ping":
                    await ws.send(json.dumps({"type": "pong", "event_id": event["ping_event"]["event_id"]}))
                elif kind == "user_transcript":
                    logger.info("you: %s", event["user_transcription_event"]["user_transcript"])
                elif kind == "agent_response":
                    logger.info("agent: %s", event["agent_response_event"]["agent_response"])

        mic = sd.RawInputStream(
            samplerate=mic_rate,
            channels=1,
            dtype="int16",
            blocksize=int(mic_rate * CHUNK_SECONDS),
            device=mic_index,
            callback=on_mic,
        )
        try:
            with mic:
                logger.info("listening; start talking")
                tasks = [asyncio.create_task(t) for t in (send_mic(), receive(), stop.wait())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        raise task.exception()  # type: ignore[misc]
        finally:
            player.close()
    logger.info("conversation ended")


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk with an ElevenLabs agent through Palmimo's mic and speaker")
    parser.add_argument("--agent-id", required=True, help="Agent id from the ElevenLabs dashboard")
    parser.add_argument("--speaker-device", default="ReSpeaker", help="Part of the speaker's card name in `aplay -l`")
    parser.add_argument("--mic-device", default="ReSpeaker", help="Part of the mic's name in `arecord -l`")
    parser.add_argument("--volume", type=float, default=1.0, help="Output gain; 1.0 is the voice's own level")
    args = parser.parse_args()

    # The Portal's log view shows stdout and stderr as they are.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        logger.error("%s is not set. Add it in the Portal's environment variables screen.", API_KEY_ENV)
        return 1
    try:
        asyncio.run(converse(args, api_key))
    except websockets.ConnectionClosed as exc:
        logger.info("conversation closed by ElevenLabs (%s)", exc.rcvd.reason if exc.rcvd else "no reason given")
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
