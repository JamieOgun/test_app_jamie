"""Coding tasks on your laptop, through the desk bridge (desk_bridge/ in this repo).

The agent gets four client tools that forward to the bridge, which runs them
with Claude Code. The robot only ever calls the bridge, never the other way
round: ``watch_events`` long-polls it to hear when a task finishes or a
terminal session of yours needs you, and main.py has the agent say so.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("coding_tools")

#: Long enough for one /events long-poll (the bridge holds it 25 s).
POLL_TIMEOUT_SECS = 35.0
RETRY_SECS = 5.0

PROMPT_LINES = (
    (
        "You can run coding tasks on the user's laptop with Claude Code: start_coding_task, coding_task_status, "
        "continue_coding_task and cancel_coding_task. Pass the user's request on in their words, in English. "
        "Tasks take minutes: after starting one, confirm in a few words and let the user go. "
        "You will be told when it finishes."
    ),
    (
        "When you report on code, never read code, file paths, commands or diffs aloud: say the outcome in one "
        "short sentence, such as 'Done, the tests pass.'"
    ),
)


@dataclass
class Announcement:
    """Something from the laptop the user should hear, e.g. a task finished."""

    kind: str  # done, failed, needs_input
    repo: str
    text: str

    def message(self) -> str:
        """What to tell the agent, so it says it in its own words."""
        what = {
            "done": "Claude Code finished",
            "failed": "Claude Code failed",
            "needs_input": "Claude Code needs the user",
        }.get(self.kind, "Update from Claude Code")
        return (
            f"(Update from the laptop, not said by the user) {what} in {self.repo}: {self.text}\n"
            "Tell the user in one short sentence, in the language you have been using."
        )


class BridgeError(Exception):
    """A reason from the bridge (or about reaching it) that the agent can read out."""


class DeskBridge:
    def __init__(self, url: str, token: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=15.0
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BridgeError(f"I can't reach the laptop's desk bridge ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            raise BridgeError(body.get("error") or f"The desk bridge said HTTP {response.status_code}.")
        return body


def tool_configs(repos: list[str]) -> list[dict[str, Any]]:
    """The coding tools as ElevenLabs client tools; *repos* go in the descriptions."""
    repo_hint = f" One of: {', '.join(repos)}." if repos else ""

    def tool(name: str, description: str, properties: dict[str, Any] | None = None, required: tuple = ()) -> dict:
        config: dict[str, Any] = {
            "type": "client",
            "name": name,
            "description": description,
            "expects_response": True,
            "response_timeout_secs": 20,
        }
        if properties:
            config["parameters"] = {
                "type": "object",
                "description": f"Arguments for {name}.",
                "properties": properties,
                "required": list(required),
            }
        return config

    return [
        tool(
            "start_coding_task",
            "Start a coding task on the user's laptop with Claude Code, which edits code, runs commands and tests "
            "by itself. Use it when the user asks for a change, fix, test, or question about their code.",
            {
                "task": {"type": "string", "description": "What to do, in the user's words, in English."},
                "repo": {"type": "string", "description": f"Which repo. Leave empty for the default.{repo_hint}"},
            },
            ("task",),
        ),
        tool("coding_task_status", "How the current or last coding task is going: its state and latest step."),
        tool(
            "continue_coding_task",
            "Send a follow-up to the last coding task, in the same session, once it has finished.",
            {"message": {"type": "string", "description": "What to tell Claude Code, in English."}},
            ("message",),
        ),
        tool("cancel_coding_task", "Stop the coding task that is running."),
    ]


def _status_text(status: dict[str, Any]) -> str:
    if status.get("state") == "idle":
        return "No coding task has run yet."
    parts = [f"state: {status['state']}", f"repo: {status['repo']}", f"elapsed: {status['elapsed_secs']}s"]
    if status["state"] == "running":
        parts.append(f"latest step: {status['progress']}")
    elif status.get("summary"):
        parts.append(f"result: {status['summary'][-800:]}")
    return "; ".join(parts)


Handler = Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]


def handlers(bridge: DeskBridge) -> dict[str, Handler]:
    """Tool name to an async function returning (result text, is_error)."""

    def safe(fn: Callable[[dict[str, Any]], Awaitable[str]]) -> Handler:
        async def run(args: dict[str, Any]) -> tuple[str, bool]:
            try:
                return await fn(args), False
            except BridgeError as exc:
                return str(exc), True

        return run

    async def start(args: dict[str, Any]) -> str:
        body = {"task": args.get("task", ""), "repo": args.get("repo") or None}
        status = await bridge.request("POST", "/tasks", json=body)
        return f"Started in {status['repo']}. You will be told when it finishes."

    async def status(args: dict[str, Any]) -> str:
        return _status_text(await bridge.request("GET", "/tasks/current"))

    async def follow_up(args: dict[str, Any]) -> str:
        status = await bridge.request("POST", "/tasks/current/messages", json={"message": args.get("message", "")})
        return f"Sent to Claude Code in {status['repo']}. You will be told when it finishes."

    async def cancel(args: dict[str, Any]) -> str:
        await bridge.request("POST", "/tasks/current/cancel")
        return "Cancelling the task."

    return {
        "start_coding_task": safe(start),
        "coding_task_status": safe(status),
        "continue_coding_task": safe(follow_up),
        "cancel_coding_task": safe(cancel),
    }


async def list_repos(bridge: DeskBridge) -> tuple[list[str], int]:
    """(allowlisted repo names, newest event id); ([], 0) if the bridge cannot be reached."""
    try:
        body = await bridge.request("GET", "/repos")
    except BridgeError as exc:
        logger.warning("desk bridge not reachable yet: %s", exc)
        return [], 0
    return list(body.get("repos", [])), int(body.get("last_event", 0))


async def watch_events(bridge: DeskBridge, out: asyncio.Queue[Announcement], after: int) -> None:
    """Long-poll the bridge for events forever, starting after event id *after*."""
    reachable = True
    while True:
        try:
            body = await bridge.request("GET", "/events", params={"after": after}, timeout=POLL_TIMEOUT_SECS)
        except BridgeError as exc:
            if reachable:
                logger.warning("lost the desk bridge, retrying: %s", exc)
                reachable = False
            await asyncio.sleep(RETRY_SECS)
            continue
        if not reachable:
            logger.info("desk bridge is back")
            reachable = True
        last = int(body.get("last_event", after))
        if last < after:
            after = 0  # the bridge restarted and numbers from 1 again
            continue
        for event in body.get("events", []):
            after = max(after, int(event["id"]))
            out.put_nowait(Announcement(event["kind"], event["repo"], event["text"]))
        after = max(after, last)
