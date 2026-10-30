"""Palmimo's actions as ElevenLabs client tools.

The tools come from the SDK's AgentToolSet, the same toolbox the devkit's
companion agent uses. Two parts:

- ``sync_tools`` registers every tool in the ElevenLabs workspace, attaches
  it to the agent, adds ``PROMPT_LINES`` to the agent's prompt, and makes the
  agent quiet when nobody is talking to it (``TURN_SETTINGS`` and the
  ``skip_turn`` system tool), and clears the agent's first message, since
  the robot is woken by name and should just listen. Nothing is changed when
  it is already current.
- ``ToolRunner`` answers the agent's ``client_tool_call`` events by running the
  tool on the robot and sending back a ``client_tool_result``. ``go_to_sleep``
  is handled here rather than on the robot: it ends the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from palmimo_sdk.agent import AgentToolSet

#: ``say`` would need a speaker of its own (the agent already talks), and
#: ``capture`` returns a photo, which a client tool result cannot carry.
EXCLUDED_TOOLS = ("say", "capture")

#: Optional arguments every motion carries but that make no sense here: the
#: agent speaks with its own voice, and ``reason`` is only for logs.
HIDDEN_ARGS = ("say", "reason")

#: How long the agent waits for a tool. The longest motions run 15 s, plus a
#: little to finish the move and settle.
RESPONSE_TIMEOUT_SECS = 25

API = "https://api.elevenlabs.io/v1/convai"

#: Added to the end of the agent's prompt (each once). The first makes it use
#: the tools while it talks; the rest keep it from talking when nobody needs it.
PROMPT_LINES = (
    "You have a robot body; use your tools to move and show feelings while you talk.",
    "Keep replies short: one or two sentences, and only ask a question when you need the answer.",
    (
        "You sit in a room where people also talk to each other. If speech is not meant for you, "
        "the user goes quiet, or there is nothing useful to add, call skip_turn and stay silent. "
        "Never ask whether the user is still there."
    ),
    (
        "People wake you by calling your name, Palmimo (パルミーモ); reply in the language they use. "
        "When they say goodbye or ask you to sleep, say a short goodbye and call go_to_sleep."
    ),
)

#: Ends the conversation and puts the robot back to listening for its name.
#: Answered by ToolRunner itself, not by the SDK toolbox.
GO_TO_SLEEP = {
    "type": "client",
    "name": "go_to_sleep",
    "description": (
        "Go back to sleep and stop listening until someone calls your name again. Call it when the user "
        "says goodbye, asks you to sleep or stop listening, or says that's all. Say a short goodbye first. "
        "Do not call it for pauses, silence, or when you are unsure."
    ),
    "expects_response": True,
    "response_timeout_secs": 5,
}

#: ElevenLabs re-engages the user after ``turn_timeout`` seconds of silence
#: (default 7). 30 is the longest it allows; skip_turn covers the rest.
TURN_SETTINGS = {"turn_timeout": 30, "turn_eagerness": "patient"}

#: The system tool that lets the agent pass a turn without speaking. With no
#: wait timeout it then stays silent until someone speaks.
SKIP_TURN = {"type": "system", "name": "skip_turn", "description": "", "params": {"system_tool_type": "skip_turn"}}

logger = logging.getLogger("robot_tools")


def build_toolset(robot: Any) -> AgentToolSet:
    return AgentToolSet(robot, exclude=EXCLUDED_TOOLS)


def _param(name: str, prop: dict[str, Any]) -> dict[str, Any]:
    """One pydantic property as an ElevenLabs literal property (no anyOf, null, or numeric bounds)."""
    for branch in prop.get("anyOf", []):
        if branch.get("type") != "null":
            prop = {**prop, **branch}
            break
    description = prop.get("description") or name.replace("_", " ")
    if "minimum" in prop and "maximum" in prop:
        description += f" Between {prop['minimum']:g} and {prop['maximum']:g}."
    if prop.get("default") is not None:
        description += f" Default {prop['default']}."
    out: dict[str, Any] = {"type": prop["type"], "description": description}
    if "enum" in prop:
        out["enum"] = prop["enum"]
    return out


def client_tool_configs(toolset: AgentToolSet) -> list[dict[str, Any]]:
    """Every tool in *toolset* as an ElevenLabs client tool config."""
    configs = []
    for name, cls in toolset.tool_models.items():
        schema = cls.parameters_schema()
        properties = {
            key: _param(key, prop) for key, prop in schema.get("properties", {}).items() if key not in HIDDEN_ARGS
        }
        config: dict[str, Any] = {
            "type": "client",
            "name": name,
            "description": cls.description,
            "expects_response": True,
            "response_timeout_secs": RESPONSE_TIMEOUT_SECS,
        }
        if properties:
            config["parameters"] = {
                "type": "object",
                "description": f"Arguments for {name}.",
                "properties": properties,
                "required": [key for key in schema.get("required", []) if key in properties],
            }
        configs.append(config)
    configs.append(GO_TO_SLEEP)
    return configs


def _request(method: str, path: str, api_key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"xi-api-key": api_key, "content-type": "application/json"}
    request = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"{method} {path} failed (HTTP {exc.code}): {detail}") from exc


def _same(existing: Any, wanted: Any) -> bool:
    """Whether *existing* already has every field of *wanted*; ElevenLabs adds defaults of its own when it stores a tool."""
    if isinstance(wanted, dict):
        return isinstance(existing, dict) and all(_same(existing.get(key), value) for key, value in wanted.items())
    return existing == wanted


def sync_tools(agent_id: str, api_key: str, configs: list[dict[str, Any]]) -> None:
    """Make sure every config exists as a workspace tool, is current, and is attached to the agent.

    Also appends any missing PROMPT_LINES to the agent's prompt, enables
    skip_turn, and applies TURN_SETTINGS.
    """
    existing = {
        tool["tool_config"]["name"]: tool
        for tool in _request("GET", "/tools", api_key).get("tools", [])
        if tool.get("tool_config", {}).get("type") == "client"
    }
    tool_ids = []
    for config in configs:
        tool = existing.get(config["name"])
        if tool is None:
            tool = _request("POST", "/tools", api_key, {"tool_config": config})
            logger.info("registered tool %s", config["name"])
        elif not _same(tool["tool_config"], config):
            _request("PATCH", f"/tools/{tool['id']}", api_key, {"tool_config": config})
            logger.info("updated tool %s", config["name"])
        tool_ids.append(tool["id"])

    agent = _request("GET", f"/agents/{agent_id}", api_key)
    config = agent["conversation_config"]
    prompt = config["agent"]["prompt"]
    attached = prompt.get("tool_ids") or []
    missing = [tool_id for tool_id in tool_ids if tool_id not in attached]
    text = prompt.get("prompt") or ""
    new_lines = [line for line in PROMPT_LINES if line not in text]
    built_in = prompt.get("built_in_tools") or {}
    needs_skip = not built_in.get("skip_turn")
    turn = config.get("turn") or {}
    needs_turn = not _same(turn, TURN_SETTINGS)
    needs_quiet_start = bool(config["agent"].get("first_message"))
    if not missing and not new_lines and not needs_skip and not needs_turn and not needs_quiet_start:
        logger.info("agent is up to date: %d robot tools attached", len(tool_ids))
        return
    # Send the whole prompt and turn config back with only these fields
    # changed, so nothing else on the agent is reset. The deprecated inline
    # `tools` list cannot be sent together with tool_ids.
    prompt = {key: value for key, value in prompt.items() if key != "tools"}
    prompt["tool_ids"] = attached + missing
    if new_lines:
        prompt["prompt"] = "\n\n".join([text.rstrip(), *new_lines]) if text.strip() else "\n\n".join(new_lines)
    if needs_skip:
        prompt["built_in_tools"] = {**built_in, "skip_turn": SKIP_TURN}
    body: dict[str, Any] = {"agent": {"prompt": prompt}}
    if needs_quiet_start:
        body["agent"]["first_message"] = ""
    if needs_turn:
        body["turn"] = {**turn, **TURN_SETTINGS}
    _request("PATCH", f"/agents/{agent_id}", api_key, {"conversation_config": body})
    logger.info(
        "updated the agent: %d tools attached, %d prompt lines added, skip_turn %s, turn settings %s, first message %s",
        len(missing),
        len(new_lines),
        "enabled" if needs_skip else "present",
        "updated" if needs_turn else "current",
        "cleared" if needs_quiet_start else "already empty",
    )


class ToolRunner:
    """Runs the agent's tool calls on the robot, one at a time, and reports each result."""

    def __init__(self, toolset: AgentToolSet, send: Any) -> None:
        self._toolset = toolset
        self._send = send  # async callable taking one JSON-able dict
        self._tasks: set[asyncio.Task[None]] = set()
        #: Set when the agent calls go_to_sleep.
        self.sleep_requested = asyncio.Event()
        #: When the last tool finished (time.monotonic()); 0 before any.
        self.last_finished = 0.0

    @property
    def busy(self) -> bool:
        """Whether a tool is still running (a long dance should not count as silence)."""
        return bool(self._tasks)

    def handle(self, call: dict[str, Any]) -> None:
        """Start one ``client_tool_call`` in the background, so audio keeps flowing while the robot moves."""
        task = asyncio.create_task(self._run(call))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, call: dict[str, Any]) -> None:
        name = call.get("tool_name", "")
        args = {key: value for key, value in (call.get("parameters") or {}).items() if key not in HIDDEN_ARGS}
        logger.info("tool %s(%s)", name, json.dumps(args))
        if name == GO_TO_SLEEP["name"]:
            text, is_error = "going to sleep", False
            self.sleep_requested.set()
        else:
            # AgentToolSet.call never raises for a bad call; it returns an error the agent can read.
            result = await self._toolset.call(name, args)
            text, is_error = result.text, result.is_error
        logger.info("  -> %s", text[:120])
        self.last_finished = time.monotonic()
        await self._send(
            {
                "type": "client_tool_result",
                "tool_call_id": call["tool_call_id"],
                "result": text,
                "is_error": is_error,
            }
        )

    async def interrupt(self) -> None:
        """Stop the motion in progress; its call still gets an "interrupted" result."""
        await self._toolset.cancel_running()

    async def close(self) -> None:
        """Cancel every tool still running and wait for the robot to stop moving."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
