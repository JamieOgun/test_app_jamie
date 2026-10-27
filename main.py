"""Palmimo as a companion robot that speaks with an ElevenLabs voice.

The robot itself is the devkit's companion agent (palmimo-companion-agent):
it listens, decides whether it was spoken to, answers through an LLM that
also picks its motions and faces, and idles on its own in between. This app
changes two things about it:

- every line is spoken by ElevenLabs instead of piper or OpenAI, with piper
  as a fallback so a network hiccup costs one line's voice, not the line;
- the respond prompt tells the LLM it can put ElevenLabs audio tags such as
  [excited] in what it says, so it chooses the delivery along with the words.

The manifest also points speech-to-text at ElevenLabs Scribe
(elevenlabs/scribe_v2), so one ElevenLabs key covers hearing and speaking.

This app also picks the audio hardware, which the companion only reads
from COMPANION_AGENT_* variables: --speaker-device and --mic-device name
the cards by part of their names, and --echo-cancel turns on echo
cancellation (it needs a 6-channel ReSpeaker; a plain USB mic does not
have the playback channel it cancels against).

Other flags go to the companion agent unchanged; run with --help to list them.
"""

import argparse
import os
import sys

from palmimo_companion_agent.main import run as run_companion
from palmimo_companion_agent.pipeline import respond, wiring
from palmimo_companion_agent.pipeline.settings import PipelineSettings
from palmimo_sdk import MicStream, PiperEngine, TtsEngine
from palmimo_sdk.audio.aec import EchoCanceller

from elevenlabs_engine import API_KEY_ENV, DEFAULT_VOICE_ID, ElevenLabsEngine

# Appended to the respond turn's system prompt. Written in Japanese like the
# companion's own prompt files. Only respond turns speak; idle turns have no say.
AUDIO_TAG_PROMPT = """\
## 声の演出（音声タグ）

- 声は ElevenLabs。`say` のテキストに英語の音声タグを角括弧で入れると、声の調子が変わる。
  タグそのものは読み上げられない。
- 使えるタグの例: `[excited]` `[happy]` `[curious]` `[surprised]` `[whispers]` `[sad]`
  `[laughs]` `[giggles]` `[sighs]`
- タグは変えたい部分の直前に置く。例: `say="[excited] みつけた！ [whispers] ひみつだよ"`
- 1 回の発話にタグは 1〜2 個まで。毎回は付けず、気持ちが動いた時だけ使う。
"""


def _patch_companion(voice_id: str, stability: float, mic_device: str | None) -> None:
    """Point the companion's voice at ElevenLabs, add audio tags to its prompt,
    and open the mic named by *mic_device*.

    These hooks are private to the companion agent, which uv.lock pins to one
    commit; check they are still there so a dependency bump fails here, loudly,
    instead of quietly running with the old voice.
    """
    for module, attr in ((wiring, "_build_engine"), (wiring, "_build_mic"), (respond, "load_prompt")):
        if not callable(getattr(module, attr, None)):
            raise RuntimeError(f"{module.__name__}.{attr} is gone; this app needs updating for the new companion agent")

    def build_engine(settings: PipelineSettings) -> TtsEngine:
        fallback = PiperEngine(
            data_dir=settings.voice_dir,
            length_scale=1.0 / settings.voice_speed,
            volume=settings.voice_volume,
        )
        return ElevenLabsEngine(
            voice_id=voice_id,
            stability=stability,
            speed=settings.voice_speed,
            volume=settings.voice_volume,
            fallback=fallback,
        )

    def build_mic(settings: PipelineSettings) -> MicStream:
        # Same as the companion's own _build_mic, plus the device hint it has no setting for.
        if not settings.echo_cancel:
            return MicStream(device_name_hint=mic_device, processors=[])
        canceller = EchoCanceller(near_channel=settings.near_channel, reference_channel=settings.reference_channel)
        return MicStream(device_name_hint=mic_device, processors=[canceller])

    original_load_prompt = respond.load_prompt

    def load_prompt(kind):  # type: ignore[no-untyped-def]
        prompt = original_load_prompt(kind)
        return f"{prompt}\n\n{AUDIO_TAG_PROMPT}" if kind == "respond" else prompt

    wiring._build_engine = build_engine
    wiring._build_mic = build_mic
    # respond.py imported load_prompt by name, so patch its copy.
    respond.load_prompt = load_prompt


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--stability", type=float, default=0.4)
    parser.add_argument("--speaker-device", default="ReSpeaker")
    parser.add_argument("--mic-device", default="ReSpeaker")
    parser.add_argument("--echo-cancel", action="store_true")
    args, companion_argv = parser.parse_known_args()

    if not os.environ.get(API_KEY_ENV):
        print(f"Error: {API_KEY_ENV} is not set. Add it in the Portal's environment variables screen.", file=sys.stderr)
        sys.exit(1)

    # The companion reads these through its pydantic settings, not its CLI.
    os.environ["COMPANION_AGENT_SPEAKER_DEVICE"] = args.speaker_device
    os.environ["COMPANION_AGENT_ECHO_CANCEL"] = "true" if args.echo_cancel else "false"

    _patch_companion(args.voice_id, args.stability, args.mic_device or None)
    sys.argv = [sys.argv[0], *companion_argv]
    run_companion()


if __name__ == "__main__":
    main()
