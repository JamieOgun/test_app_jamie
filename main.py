"""Speak a line on Palmimo's speaker with an ElevenLabs voice.

The text can carry ElevenLabs v4 audio tags, such as "[excited] Hello!",
which change how the line is delivered instead of being read out.
"""

import argparse
import logging
import os
import sys

from palmimo_sdk import Speaker, SpeakerConfig

from elevenlabs_engine import API_KEY_ENV, DEFAULT_VOICE_ID, ElevenLabsEngine

logger = logging.getLogger("test_app_jamie")


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak text with an ElevenLabs voice")
    parser.add_argument("--text", default="[excited] Hello! I'm Palmimo.")
    parser.add_argument("--lang", choices=["en", "ja"], default="en")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--volume", type=float, default=1.0)
    args = parser.parse_args()

    # The Portal's log view shows stdout and stderr as they are.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.environ.get(API_KEY_ENV):
        logger.error("%s is not set. Add it in the Portal's environment variables screen.", API_KEY_ENV)
        return 1

    engine = ElevenLabsEngine(voice_id=args.voice_id, speed=args.speed, volume=args.volume)
    with Speaker(SpeakerConfig(lang=args.lang), engine=engine) as speaker:
        logger.info("speaking (%s, voice %s): %s", args.lang, args.voice_id, args.text)
        handle = speaker.say(args.text)
        handle.join()
        if handle.error is not None:
            logger.error("speech failed: %s", handle.error)
            return 1
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
