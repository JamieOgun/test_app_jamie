"""Runs one Claude Code task at a time through the Claude Agent SDK.

Tasks run with every permission auto-approved (``bypassPermissions``): nobody
is at the laptop to answer prompts, and voice approval would let anyone in the
room say yes. The repo allowlist, the token and the budget cap in the bridge's
config are the guardrails.

``Runner`` is the small interface the server uses, so a Codex runner
(``codex exec --json``) can be added beside this one later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

logger = logging.getLogger("desk_bridge.runner")

#: Appended to Claude Code's own system prompt: the result is spoken by the robot.
SPOKEN_SUMMARY = (
    "You were given this task by voice through Palmimo, a desk robot, and nobody is watching the terminal. "
    "Do the whole task without asking questions; make reasonable choices and mention them. "
    "End your final message with one short plain sentence that says the outcome, written to be spoken aloud: "
    "no code, file paths, markdown or lists."
)


class Busy(Exception):
    """A task is already running."""


@dataclass
class Task:
    task: str
    repo: str
    state: str = "running"  # running, done, failed, cancelled
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    progress: str = "starting"
    summary: str = ""
    session_id: str | None = None
    cost_usd: float | None = None
    cancel_requested: bool = False

    def status(self) -> dict[str, Any]:
        end = self.ended or time.monotonic()
        return {
            "state": self.state,
            "task": self.task,
            "repo": self.repo,
            "elapsed_secs": round(end - self.started),
            "progress": self.progress,
            "summary": self.summary,
            "cost_usd": self.cost_usd,
        }


#: Called with (kind, repo, text) when a task ends; kind is "done" or "failed".
Notify = Callable[[str, str, str], Awaitable[None]]


class Runner(Protocol):
    session_ids: set[str]

    async def start(self, task: str, repo: str, cwd: str, *, follow_up: bool = False) -> Task: ...

    async def cancel(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...


def describe_tool(block: ToolUseBlock) -> str:
    """A short progress line for a tool call, e.g. "running Bash: pytest -q"."""
    detail = block.input.get("command") or block.input.get("file_path") or block.input.get("pattern") or ""
    detail = str(detail).splitlines()[0][:80] if detail else ""
    return f"running {block.name}: {detail}" if detail else f"running {block.name}"


class ClaudeRunner:
    def __init__(self, notify: Notify, max_budget_usd: float, model: str | None = None) -> None:
        self._notify = notify
        self._max_budget_usd = max_budget_usd
        self._model = model
        self._current: Task | None = None
        self._job: asyncio.Task[None] | None = None
        self._client: ClaudeSDKClient | None = None
        #: Every session this runner started, so hook calls from them are not announced twice.
        self.session_ids: set[str] = set()

    @property
    def busy(self) -> bool:
        return self._current is not None and self._current.state == "running"

    def status(self) -> dict[str, Any]:
        return self._current.status() if self._current else {"state": "idle"}

    async def start(self, task: str, repo: str, cwd: str, *, follow_up: bool = False) -> Task:
        """Start *task* in *cwd*; with *follow_up*, continue the last task's session instead of a fresh one."""
        if self.busy:
            raise Busy
        resume = self._current.session_id if follow_up and self._current else None
        self._current = Task(task=task, repo=repo)
        options = ClaudeAgentOptions(
            cwd=cwd,
            permission_mode="bypassPermissions",
            setting_sources=["user", "project", "local"],
            system_prompt={"type": "preset", "preset": "claude_code", "append": SPOKEN_SUMMARY},
            max_budget_usd=self._max_budget_usd,
            model=self._model,
            resume=resume,
        )
        self._job = asyncio.create_task(self._run(self._current, options))
        logger.info("started in %s%s: %s", repo, " (follow-up)" if resume else "", task)
        return self._current

    async def cancel(self) -> bool:
        if not self.busy or self._client is None:
            return False
        self._current.cancel_requested = True
        await self._client.interrupt()
        return True

    async def _run(self, task: Task, options: ClaudeAgentOptions) -> None:
        last_text = ""
        try:
            async with ClaudeSDKClient(options=options) as client:
                self._client = client
                await client.query(task.task)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        if message.session_id:
                            self._remember(task, message.session_id)
                        for block in message.content:
                            if isinstance(block, TextBlock) and block.text.strip():
                                last_text = block.text.strip()
                                task.progress = last_text.splitlines()[-1][:160]
                            elif isinstance(block, ToolUseBlock):
                                task.progress = describe_tool(block)
                    elif isinstance(message, ResultMessage):
                        self._remember(task, message.session_id)
                        task.cost_usd = message.total_cost_usd
                        task.summary = (message.result or last_text).strip()
                        if task.cancel_requested or message.terminal_reason in ("aborted_streaming", "aborted_tools"):
                            task.state = "cancelled"
                            task.summary = "Cancelled."
                        else:
                            task.state = "failed" if message.is_error else "done"
                        if message.is_error and not task.summary:
                            task.summary = "; ".join(message.errors or []) or message.subtype
        except Exception as exc:  # noqa: BLE001 -- report every failure to the robot rather than crash the bridge
            logger.exception("task failed")
            task.state = "failed"
            task.summary = f"Claude Code stopped with an error: {exc}"
        finally:
            self._client = None
            if task.state == "running":
                task.state = "failed"
                task.summary = task.summary or "Claude Code ended without a result."
            task.ended = time.monotonic()
        logger.info("%s in %s after %ds: %s", task.state, task.repo, task.ended - task.started, task.summary[-200:])
        if task.state != "cancelled":
            await self._notify(task.state, task.repo, task.summary)

    def _remember(self, task: Task, session_id: str | None) -> None:
        if session_id:
            task.session_id = session_id
            self.session_ids.add(session_id)
