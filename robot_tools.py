"""Palmimo's actions as ElevenLabs client tools.

The tools come from the SDK's AgentToolSet, the same toolbox the devkit's
companion agent uses. Two parts:

- ``sync_tools`` registers every tool in the ElevenLabs workspace, attaches
  it to the agent, and adds ``BODY_PROMPT`` to the agent's prompt, so the
  agent's LLM knows it can move. Nothing is changed when it is already current.
- ``ToolRunner`` answers the agent's ``client_tool_call`` events by running the
  tool on the robot and sending back a ``client_tool_result``.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

#: Added to the end of the agent's prompt (once), so it uses the tools while it talks.
BODY_PROMPT = "You have a robot body; use your tools to move and show feelings while you talk."

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

    Also appends BODY_PROMPT to the agent's prompt if it is not there yet.
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
    prompt = agent["conversation_config"]["agent"]["prompt"]
    attached = prompt.get("tool_ids") or []
    missing = [tool_id for tool_id in tool_ids if tool_id not in attached]
    text = prompt.get("prompt") or ""
    needs_line = BODY_PROMPT not in text
    if not missing and not needs_line:
        logger.info("agent is up to date: %d robot tools attached", len(tool_ids))
        return
    # Send the whole prompt back with only these fields changed, so nothing
    # else on the agent is reset. The deprecated inline `tools` list cannot be
    # sent together with tool_ids.
    prompt = {key: value for key, value in prompt.items() if key != "tools"}
    prompt["tool_ids"] = attached + missing
    if needs_line:
        prompt["prompt"] = f"{text.rstrip()}\n\n{BODY_PROMPT}" if text.strip() else BODY_PROMPT
    _request("PATCH", f"/agents/{agent_id}", api_key, {"conversation_config": {"agent": {"prompt": prompt}}})
    logger.info("updated the agent: %d tools attached, body prompt %s", len(missing), "added" if needs_line else "present")


class ToolRunner:
    """Runs the agent's tool calls on the robot, one at a time, and reports each result."""

    def __init__(self, toolset: AgentToolSet, send: Any) -> None:
        self._toolset = toolset
        self._send = send  # async callable taking one JSON-able dict
        self._tasks: set[asyncio.Task[None]] = set()

    def handle(self, call: dict[str, Any]) -> None:
        """Start one ``client_tool_call`` in the background, so audio keeps flowing while the robot moves."""
        task = asyncio.create_task(self._run(call))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, call: dict[str, Any]) -> None:
        name = call.get("tool_name", "")
        args = {key: value for key, value in (call.get("parameters") or {}).items() if key not in HIDDEN_ARGS}
        logger.info("tool %s(%s)", name, json.dumps(args))
        # AgentToolSet.call never raises for a bad call; it returns an error the agent can read.
        result = await self._toolset.call(name, args)
        logger.info("  -> %s", result.text[:120])
        await self._send(
            {
                "type": "client_tool_result",
                "tool_call_id": call["tool_call_id"],
                "result": result.text,
                "is_error": result.is_error,
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
