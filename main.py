"""Talk with an ElevenLabs agent through Palmimo's mic and speaker, woken by name.

The robot sleeps until someone calls it "Palmimo" (パルミーモ); see wake.py
for how the name is heard. It then plays a chime and opens a conversation. If
something was said in the same breath ("Palmimo, dance"), that is sent to the
agent straight away. The conversation ends, and the robot goes back to
listening for its name, when the agent calls ``go_to_sleep``, after
``--idle-timeout-secs`` without activity, or at ``--max-session-secs``. So no
paid conversation stays open while nobody is talking to it.

The agent (its voice, prompt, language and LLM) is set up in the ElevenLabs
dashboard. The mic is streamed to the agent over its WebSocket, and the agent's
voice is streamed to the speaker through aplay.

The agent can also move the robot: every action in the SDK's toolbox (walk,
turn, dance, nod, look, show a face, ...) is registered on the agent as a
client tool when the app starts, and its calls run on the robot here. See
robot_tools.py.

With ``--coding-workdir`` set, the agent can also run coding tasks with Codex
on the repos cloned into that folder on the robot. When a task finishes, the
robot wakes by itself (if asleep) and the agent says so. See coding_tools.py.

The mic is muted while the robot is talking. Without a ReSpeaker there is no
echo cancellation, so the agent would otherwise hear its own voice and
interrupt itself. The cost is that you cannot talk over it.

The app runs until the Portal stops it.
"""

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
from openai import OpenAI
from palmimo_sdk.agent import AgentToolSet
from palmimo_sdk.io.alsa_devices import resolve_alsa_device
from palmimo_sdk.robot import Palmimo

import coding_tools
from coding_tools import Announcement, CodexRunner
from robot_tools import PROMPT_LINES, ToolRunner, build_toolset, client_tool_configs, sync_tools
from wake import SAMPLE_RATE as WAKE_RATE
from wake import NameWakeDetector, WakeDetector, WakeEvent

API_KEY_ENV = "ELEVENLABS_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
SIGNED_URL = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}"

#: Mic chunk length sent to the agent.
CHUNK_SECONDS = 0.05
#: Keep the mic muted this long after the robot's last sample, to cover the
#: speaker's own buffering and room echo.
MUTE_TAIL_SECONDS = 0.4
#: The wake chime: two notes of this length, at this rate.
CHIME_NOTE_SECONDS = 0.09
CHIME_RATE = 16000
#: After go_to_sleep, wait at most this long for the goodbye to finish playing.
SLEEP_GOODBYE_MAX_SECONDS = 8.0

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


def find_device(hint: str | None, kind: str = "input") -> int | None:
    """Index of the first *kind* ("input" or "output") device whose name contains *hint*, or None for the default."""
    if not hint:
        return None
    channels = f"max_{kind}_channels"
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device[channels] > 0 and hint.lower() in device["name"].lower():
            logger.info("Selected %s %r for hint %r", kind, device["name"], hint)
            return index
    names = [d["name"] for d in devices if d[channels] > 0]
    logger.warning("No %s device matches hint %r; using the default. Found: %s", kind, hint, names)
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


def open_robot(with_hardware: bool) -> contextlib.AbstractContextManager[Palmimo]:
    """The robot's legs, neck and face; no speaker or mic, since this app streams audio itself.

    Entering wakes the robot and leaving parks it, even after an error. Without
    hardware the tools still run, but nothing moves.
    """
    if not with_hardware:
        return contextlib.nullcontext(Palmimo())
    from palmimo_sdk.io.display import FaceDisplay
    from palmimo_sdk.io.dynamixel import DynamixelDriver

    return Palmimo(driver=DynamixelDriver(), display=FaceDisplay())


class SoundDevicePlayer:
    """The same interface as Player, through sounddevice, for machines without aplay (such as a Mac)."""

    def __init__(self, device: int | None, rate: int, volume: float) -> None:
        self._rate = rate
        self._volume = volume
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.busy_until = 0.0
        self._stream = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", device=device, callback=self._fill
        )
        self._stream.start()

    def _fill(self, out, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        with self._lock:
            n = min(len(out), len(self._buffer))
            out[:n] = self._buffer[:n]
            del self._buffer[:n]
        out[n:] = b"\x00" * (len(out) - n)

    def play(self, pcm: bytes) -> None:
        if self._volume != 1.0:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * self._volume
            pcm = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
        self.busy_until = max(time.monotonic(), self.busy_until) + len(pcm) / 2 / self._rate
        with self._lock:
            self._buffer += pcm

    def interrupt(self) -> None:
        with self._lock:
            self._buffer.clear()
        self.busy_until = time.monotonic()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def open_player(hint: str | None, rate: int, volume: float) -> Player | SoundDevicePlayer:
    """aplay on the robot (it picks the ALSA card reliably); sounddevice where aplay does not exist."""
    if shutil.which("aplay"):
        return Player(resolve_alsa_device(hint, kind="playback", log=logger), rate, volume)
    return SoundDevicePlayer(find_device(hint, "output"), rate, volume)


async def send_context(ws: websockets.ClientConnection, text: str) -> None:
    """Tell the agent about something that happened around the robot, e.g. "Someone just waved at you."

    The agent takes it into account on its next reply; it is not treated as
    something the user said and does not make the agent speak by itself.
    """
    await ws.send(json.dumps({"type": "contextual_update", "text": text}))


def chime(rate: int) -> bytes:
    """Two short rising notes, played the moment the robot hears its name."""
    notes = []
    for freq in (880.0, 1320.0):
        t = np.arange(int(rate * CHIME_NOTE_SECONDS)) / rate
        fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fades, no clicks
        notes.append(0.3 * np.sin(2 * np.pi * freq * t) * fade)
    return (np.concatenate(notes) * 32767).astype("<i2").tobytes()


async def play_chime(args: argparse.Namespace) -> None:
    player = open_player(args.speaker_device or None, CHIME_RATE, args.volume)
    try:
        player.play(chime(CHIME_RATE))
        await asyncio.sleep(max(0.0, player.busy_until - time.monotonic()) + 0.15)
    finally:
        player.close()


async def show_face(robot: Palmimo, expression: str | None) -> None:
    """Show *expression* on the face display, or the neutral face for None; skipped without a display."""
    display = robot.display
    if display is None or not display.is_connected:
        return
    try:
        if expression is None:
            await asyncio.to_thread(display.idle)
        else:
            await asyncio.to_thread(display.set_expression, expression)
    except Exception as exc:  # noqa: BLE001 --  # the face is decoration; never let it stop the voice
        logger.warning("could not change the face: %s", exc)


def drain(chunks: asyncio.Queue[bytes]) -> None:
    while not chunks.empty():
        chunks.get_nowait()


async def wait_for_wake(
    detector: WakeDetector,
    mic_chunks: asyncio.Queue[bytes],
    mic_rate: int,
    announcements: asyncio.Queue[Announcement],
    stop: asyncio.Event,
) -> WakeEvent | Announcement | None:
    """Feed the mic to *detector* until it hears the robot's name or the laptop has news; None when stopping."""
    while not stop.is_set():
        if not announcements.empty():
            return announcements.get_nowait()
        try:
            raw = await asyncio.wait_for(mic_chunks.get(), timeout=0.5)
        except TimeoutError:
            continue
        event = await detector.feed(resample(raw, mic_rate, WAKE_RATE))
        if event is not None:
            return event
    return None


async def conversation(
    args: argparse.Namespace,
    api_key: str,
    toolset: AgentToolSet,
    mic_chunks: asyncio.Queue[bytes],
    mic_rate: int,
    woken: WakeEvent | Announcement,
    handlers: dict[str, coding_tools.Handler],
    announcements: asyncio.Queue[Announcement],
    stop: asyncio.Event,
) -> str:
    """One ElevenLabs conversation, from wake to sleep. Returns why it ended.

    *woken* is what started it: the name (maybe with a command said in the
    same breath) or news from the laptop. Further news from the laptop is
    passed on while it runs.
    """
    url = await asyncio.to_thread(get_signed_url, args.agent_id, api_key)
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

        player = open_player(args.speaker_device or None, out_rate, args.volume)

        async def send(event: dict) -> None:
            await ws.send(json.dumps(event))

        tools = ToolRunner(toolset, send, handlers)
        started = last_heard = last_audio = time.monotonic()

        async def announce(news: Announcement) -> None:
            nonlocal last_heard
            last_heard = time.monotonic()
            logger.info("codex: %s in %s: %s", news.kind, news.repo, news.text[:120])
            await send({"type": "user_message", "text": news.message()})

        if isinstance(woken, Announcement):
            await announce(woken)
        elif woken.command:
            # Said in the same breath as the name: answer it straight away.
            logger.info("you: %s", woken.command)
            await send({"type": "user_message", "text": woken.command})

        async def forward_announcements() -> None:
            while True:
                await announce(await announcements.get())

        async def send_mic() -> None:
            while True:
                pcm = resample(await mic_chunks.get(), mic_rate, in_rate)
                if time.monotonic() < player.busy_until + MUTE_TAIL_SECONDS:
                    pcm = np.zeros_like(pcm)  # muted while the robot talks
                await send({"user_audio_chunk": base64.b64encode(pcm.tobytes()).decode()})

        async def receive() -> str:
            nonlocal last_heard, last_audio
            async for message in ws:
                event = json.loads(message)
                kind = event.get("type")
                if kind == "audio":
                    last_audio = time.monotonic()
                    player.play(base64.b64decode(event["audio_event"]["audio_base_64"]))
                elif kind == "client_tool_call":
                    tools.handle(event["client_tool_call"])
                elif kind == "interruption":
                    player.interrupt()
                    await tools.interrupt()
                elif kind == "ping":
                    await send({"type": "pong", "event_id": event["ping_event"]["event_id"]})
                elif kind == "user_transcript":
                    last_heard = time.monotonic()
                    logger.info("you: %s", event["user_transcription_event"]["user_transcript"])
                elif kind == "agent_response":
                    logger.info("agent: %s", event["agent_response_event"]["agent_response"])
            return "closed by ElevenLabs"

        async def watch() -> str:
            """Decide when to go back to sleep."""
            sleep_asked = 0.0
            while True:
                await asyncio.sleep(0.25)
                now = time.monotonic()
                if tools.sleep_requested.is_set():
                    sleep_asked = sleep_asked or now
                    # Let the goodbye finish: no new audio for a moment and nothing left to play.
                    goodbye_done = now > max(player.busy_until + 0.3, last_audio + 1.0)
                    if goodbye_done or now - sleep_asked > SLEEP_GOODBYE_MAX_SECONDS:
                        return "asked to sleep"
                if now - started > args.max_session_secs:
                    return f"session cap of {args.max_session_secs:g}s"
                last_activity = max(last_heard, player.busy_until, tools.last_finished)
                if not tools.busy and now - last_activity > args.idle_timeout_secs:
                    return f"idle for {args.idle_timeout_secs:g}s"

        logger.info("listening; start talking")
        tasks = {asyncio.create_task(t) for t in (send_mic(), receive(), watch(), forward_announcements(), stop.wait())}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            task = done.pop()
            if task.exception() is not None:
                exc = task.exception()
                if isinstance(exc, websockets.ConnectionClosed):
                    return "closed by ElevenLabs"
                raise exc  # type: ignore[misc]
            result = task.result()
            return result if isinstance(result, str) else "app stopping"
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await tools.close()
            player.close()


async def run(args: argparse.Namespace, api_key: str, robot: Palmimo) -> None:
    """Sleep until someone calls the robot's name, talk, go back to sleep; repeat until stopped."""
    toolset = build_toolset(robot)
    configs = client_tool_configs(toolset)
    prompt_lines = PROMPT_LINES
    announcements: asyncio.Queue[Announcement] = asyncio.Queue()
    handlers: dict[str, coding_tools.Handler] = {}
    codex: CodexRunner | None = None
    if args.coding_workdir:
        workdir = Path(args.coding_workdir).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        codex = CodexRunner(
            workdir, announcements, model=args.codex_model, sandbox=args.codex_sandbox, network=not args.no_codex_network
        )
        repos = codex.repos()
        logger.info("coding with Codex in %s (repos: %s)", workdir, ", ".join(repos) or "none; git clone some there")
        configs += coding_tools.tool_configs(repos)
        prompt_lines = PROMPT_LINES + coding_tools.PROMPT_LINES
        handlers = coding_tools.handlers(codex)
    try:
        await asyncio.to_thread(sync_tools, args.agent_id, api_key, configs, prompt_lines)
    except (RuntimeError, OSError, KeyError) as exc:
        # The conversation still works without them; the agent just cannot move the robot.
        logger.warning("could not register the robot's tools on the agent: %s", exc)

    # Loads (and on first run downloads) the speech detector.
    detector: WakeDetector = await asyncio.to_thread(NameWakeDetector, OpenAI(), args.stt_model)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    # One mic stream for the whole run, at its own rate (cheap USB mics often
    # cannot open at 16 kHz); each consumer resamples what it needs.
    mic_index = find_device(args.mic_device or None)
    mic_rate = int(sd.query_devices(mic_index, "input")["default_samplerate"])
    mic_chunks: asyncio.Queue[bytes] = asyncio.Queue()

    def on_mic(data, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        loop.call_soon_threadsafe(mic_chunks.put_nowait, bytes(data))

    mic = sd.RawInputStream(
        samplerate=mic_rate,
        channels=1,
        dtype="int16",
        blocksize=int(mic_rate * CHUNK_SECONDS),
        device=mic_index,
        callback=on_mic,
    )
    with mic:
        while not stop.is_set():
            await show_face(robot, "SLEEP")
            logger.info("asleep; call me by name (Palmimo / パルミーモ)")
            woken = await wait_for_wake(detector, mic_chunks, mic_rate, announcements, stop)
            if woken is None:
                break
            if isinstance(woken, Announcement):
                logger.info("waking up for news from the laptop")
            await show_face(robot, None)
            await play_chime(args)
            drain(mic_chunks)  # don't send the chime itself to the agent
            try:
                reason = await conversation(
                    args, api_key, toolset, mic_chunks, mic_rate, woken, handlers, announcements, stop
                )
            except (RuntimeError, OSError, websockets.WebSocketException) as exc:
                logger.warning("could not talk to ElevenLabs: %s", exc)
                reason = "error"
            logger.info("conversation ended (%s)", reason)
            detector.reset()
            drain(mic_chunks)
    if codex is not None:
        await codex.close()
    logger.info("stopped")


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk with an ElevenLabs agent through Palmimo's mic and speaker")
    parser.add_argument("--agent-id", required=True, help="Agent id from the ElevenLabs dashboard")
    parser.add_argument("--speaker-device", default="ReSpeaker", help="Part of the speaker's card name in `aplay -l`")
    parser.add_argument("--mic-device", default="ReSpeaker", help="Part of the mic's name in `arecord -l`")
    parser.add_argument("--volume", type=float, default=1.0, help="Output gain; 1.0 is the voice's own level")
    parser.add_argument(
        "--idle-timeout-secs", type=float, default=25.0, help="Go back to sleep after this long without activity"
    )
    parser.add_argument("--max-session-secs", type=float, default=600.0, help="Longest one conversation may run")
    parser.add_argument(
        "--stt-model", default="gpt-4o-mini-transcribe", help="OpenAI model that transcribes speech to hear the name"
    )
    parser.add_argument(
        "--coding-workdir", default="", help="Folder of git repos Codex may work in, e.g. ~/code; empty for no coding"
    )
    parser.add_argument("--codex-model", default="", help="Codex model; empty for Codex's default")
    parser.add_argument(
        "--codex-sandbox",
        default="workspace-write",
        choices=("read-only", "workspace-write", "full-access"),
        help="What Codex may touch: read-only, workspace-write (only the repo), or full-access",
    )
    parser.add_argument(
        "--no-codex-network", action="store_true", help="Block network access for Codex's commands (no git push)"
    )
    parser.add_argument(
        "--no-robot", action="store_true", help="Run the tools without motors or face display (for testing off the robot)"
    )
    args = parser.parse_args()

    # The Portal's log view shows stdout and stderr as they are.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per transcription otherwise

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        logger.error("%s is not set. Add it in the Portal's environment variables screen.", API_KEY_ENV)
        return 1
    if not os.environ.get(OPENAI_KEY_ENV):
        logger.error("%s is not set. Add it in the Portal's environment variables screen.", OPENAI_KEY_ENV)
        return 1
    try:
        with open_robot(not args.no_robot) as robot:
            asyncio.run(run(args, api_key, robot))
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
