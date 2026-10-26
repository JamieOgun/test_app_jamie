# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jizai Inc.
"""ElevenLabsEngine — ElevenLabs' speech API as a :class:`~palmimo_sdk.io.tts.base.TtsEngine`.

What this buys over the other engines is delivery. The default model,
``eleven_v4_turbo``, is a real-time model that reads *audio tags* inline in
the text -- ``[excited] No way!``, ``[whispers] it's a secret`` -- so an LLM
can say how a line should sound in the same string it writes the line in, with
no separate channel between the chat model and the voice. Any other engine
would read those tags out loud, so :func:`strip_audio_tags` removes them
before text reaches a voice that does not understand them.

A network voice can fail mid-conversation (venue Wi-Fi, an exhausted quota).
With a ``fallback`` engine attached, a failed request is spoken by that engine
instead -- tags stripped -- so the robot sounds worse for a line rather than
going silent. The failure is still logged.

Like :class:`~palmimo_sdk.io.tts.openai.OpenAiEngine` there is no model to
download and nothing to load, so :meth:`ElevenLabsEngine.load_voice` is cheap
and :meth:`ElevenLabsEngine.preflight` only checks the API key.

Uses ``urllib`` rather than the ``elevenlabs`` client: one POST returning raw
PCM does not justify a dependency, and :class:`~palmimo_sdk.io.speaker.Speaker`
already calls this on its own worker thread, so blocking here is correct.
"""

from __future__ import annotations

import array
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from typing import Any

from palmimo_sdk.io.tts import TtsEngine, TtsVoice


logger = logging.getLogger(__name__)

SPEECH_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

#: Environment variable holding the API key.
API_KEY_ENV = "ELEVENLABS_API_KEY"

#: ElevenLabs' real-time v4 model, and the v4 model that reads audio tags.
#: ``"eleven_v4"`` sounds richer but is not real-time and ignores the tags.
DEFAULT_MODEL = "eleven_v4_turbo"

#: A stock ElevenLabs voice so a first run works with only a key. Pick a better
#: fit for the character from the ElevenLabs voice library and pass its id.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

#: Raw 16-bit little-endian mono PCM at this rate -- no decoder needed to put
#: it in a WAV.
SAMPLE_RATE_HZ = 24000
_OUTPUT_FORMAT = f"pcm_{SAMPLE_RATE_HZ}"

#: Short on purpose: well under Speaker's own say_timeout_s, so a slow network
#: turns into a fallback line (or a synthesis failure) instead of the whole
#: utterance timing out.
DEFAULT_TIMEOUT_S = 6.0

#: ``[word]`` or ``[a few words]`` -- the audio-tag shape. Bounded so a stray
#: bracketed sentence is not mistaken for a tag.
_AUDIO_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z '\-]{0,30}\]\s*")


def strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs audio tags (``[excited]``, ``[laughs]``, ...) from *text*.

    Args:
        text (str): An utterance that may carry audio tags.

    Returns:
        str: The utterance with tags removed and whitespace collapsed.
    """
    return " ".join(_AUDIO_TAG_RE.sub(" ", text).split())


#: ``(voice_id, body, api_key, timeout) -> PCM bytes``.
Transport = Callable[[str, dict[str, Any], str, float], bytes]


def _default_transport(voice_id: str, body: dict[str, Any], api_key: str, timeout: float) -> bytes:
    """POST *body* to the speech endpoint for *voice_id* and return the PCM bytes.

    Raises:
        RuntimeError: The API rejected the request, with its own message
            attached, or could not be reached at all.
        TimeoutError: The request ran past *timeout*.
    """
    url = SPEECH_URL.format(voice_id=voice_id) + f"?output_format={_OUTPUT_FORMAT}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "xi-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"ElevenLabs API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        # Same split as OpenAiEngine: a connect timeout arrives wrapped in
        # URLError, and is surfaced as TimeoutError so it reads like a read one.
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(f"the ElevenLabs API timed out after {timeout:g}s") from exc
        raise RuntimeError(f"ElevenLabs API is unreachable: {exc.reason}") from exc


def _scaled(pcm: bytes, volume: float) -> bytes:
    """Apply *volume* to 16-bit little-endian PCM, since the API has no gain parameter.

    Uses the stdlib ``array`` rather than numpy so this engine stays
    dependency-free; a few seconds of 24 kHz audio is cheap to scale in Python.
    """
    samples = array.array("h", pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    for i, sample in enumerate(samples):
        samples[i] = max(-32768, min(32767, round(sample * volume)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def _set_format(wav_file: wave.Wave_write) -> None:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE_HZ)


class _ElevenLabsVoice:
    """:class:`~palmimo_sdk.io.tts.base.TtsVoice` over one ElevenLabs voice, with
    an optional fallback voice behind it."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str,
        language: str,
        stability: float,
        speed: float,
        volume: float,
        timeout: float,
        transport: Transport,
        fallback: TtsVoice | None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._language = language
        self._stability = stability
        self._speed = speed
        self._volume = volume
        self._timeout = timeout
        self._transport = transport
        self._fallback = fallback

    def synthesize(self, text: str, wav_file: wave.Wave_write) -> None:
        body: dict[str, Any] = {
            "text": text,
            "model_id": self._model,
            "language_code": self._language,
            "voice_settings": {"stability": self._stability, "speed": self._speed},
        }
        try:
            pcm = self._transport(self._voice_id, body, self._api_key, self._timeout)
            if not pcm or len(pcm) % 2:
                raise RuntimeError(f"ElevenLabs API returned {len(pcm)} bytes, not 16-bit PCM")
        except Exception as exc:
            if self._fallback is None:
                # Give the writer a valid header so closing it while unwinding
                # does not replace this error with "# channels not specified".
                _set_format(wav_file)
                raise
            logger.warning("ElevenLabs failed (%s); speaking this line with the fallback voice", exc)
            self._fallback.synthesize(text, wav_file)
            return
        _set_format(wav_file)
        wav_file.writeframes(_scaled(pcm, self._volume) if self._volume != 1.0 else pcm)


class _TagStrippingVoice:
    """Wraps a voice that would read audio tags aloud, removing them first."""

    def __init__(self, inner: TtsVoice) -> None:
        self._inner = inner

    def synthesize(self, text: str, wav_file: wave.Wave_write) -> None:
        self._inner.synthesize(strip_audio_tags(text), wav_file)


class ElevenLabsEngine(TtsEngine):
    """:class:`~palmimo_sdk.io.tts.base.TtsEngine` over ElevenLabs' speech API.

    Every language uses the same voice: the model is multilingual, and the
    ``lang`` Speaker passes is sent as the request's ``language_code``.

    Args:
        voice_id (str): An ElevenLabs voice id, from the voice library.
        model (str): ElevenLabs model id; the default reads audio tags.
        stability (float): 0-1. Lower is more expressive and varies between
            takes, higher is steadier. Low suits a lively character.
        speed (float): Speaking rate, 1.0 natural.
        volume (float): Output gain applied to the returned samples, 1.0 the
            voice's own level. The API has no gain parameter. The fallback
            engine keeps its own volume setting.
        fallback (TtsEngine | None): An engine that speaks a line when the API
            fails, and every line when no API key is set. ``None`` makes a
            failure a failed utterance, like any other engine error.
        timeout_s (float): One request's budget.
        api_key_env (str): Environment variable holding the key.
        transport (Transport | None): Test seam; defaults to a real POST.
    """

    name = "elevenlabs"

    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        stability: float = 0.4,
        speed: float = 1.0,
        volume: float = 1.0,
        fallback: TtsEngine | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key_env: str = API_KEY_ENV,
        transport: Transport | None = None,
    ) -> None:
        self._voice_id = voice_id
        self._model = model
        self._stability = stability
        self._speed = speed
        self._volume = volume
        self._fallback = fallback
        self._timeout = timeout_s
        self._api_key_env = api_key_env
        self._transport = transport or _default_transport

    def preflight(self, lang: str, *, fetch: bool = True) -> None:
        """Check the key, and ready the fallback voice so a failover is instant.

        Never spends a request validating the key. ``fetch`` is passed on to
        the fallback, which may have a model to download.

        Raises:
            RuntimeError: No key and no fallback -- there is nothing to speak with.
        """
        if self._fallback is not None:
            self._fallback.preflight(lang, fetch=fetch)
        if not os.environ.get(self._api_key_env):
            if self._fallback is None:
                raise RuntimeError(f"{self._api_key_env} is not set; the ElevenLabs voice needs it.")
            logger.warning("%s is not set; speaking with the %s voice only", self._api_key_env, self._fallback.name)

    def load_voice(self, lang: str) -> TtsVoice:
        """Return a voice for *lang*, loading the fallback's voice alongside it.

        With no API key, returns the fallback's own voice, still wrapped so
        audio tags are stripped before it speaks.

        Raises:
            RuntimeError: No key and no fallback.
        """
        fallback = _TagStrippingVoice(self._fallback.load_voice(lang)) if self._fallback is not None else None
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            if fallback is None:
                raise RuntimeError(f"{self._api_key_env} is not set; the ElevenLabs voice needs it.")
            return fallback
        return _ElevenLabsVoice(
            api_key=api_key,
            voice_id=self._voice_id,
            model=self._model,
            language=lang,
            stability=self._stability,
            speed=self._speed,
            volume=self._volume,
            timeout=self._timeout,
            transport=self._transport,
            fallback=fallback,
        )

    def failure_hint(self, error_text: str, lang: str) -> str:
        """Name the failures that have a fix: a bad key, an unknown voice, no network."""
        if "401" in error_text:
            return f" {self._api_key_env} was rejected; check the key in .env"
        if "404" in error_text or "voice_not_found" in error_text:
            return f" Voice id {self._voice_id!r} was not found; copy the id from the ElevenLabs voice library"
        # Before the timeout branch, as in OpenAiEngine: "no network" is the
        # more useful hint when both could match.
        if "unreachable" in error_text:
            return " The ElevenLabs API could not be reached; this voice needs a network, unlike PiperEngine"
        if "timed out" in error_text:
            return " The ElevenLabs API did not answer in time; this voice needs a working network"
        return ""


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE_ID",
    "SPEECH_URL",
    "ElevenLabsEngine",
    "strip_audio_tags",
]
