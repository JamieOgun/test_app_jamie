"""The desk bridge: a small HTTP server on your laptop that Palmimo talks to.

The robot starts and follows coding tasks with the ``/tasks`` routes and
long-polls ``/events`` to hear when one finishes. Claude Code sessions you run
yourself in a terminal report through ``/hook`` (see README.md), so Palmimo can
announce those too. Every route except ``/hook`` needs the shared token;
``/hook`` only answers calls from this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import time
import tomllib
from pathlib import Path
from typing import Any

from aiohttp import web

from desk_bridge.runner import Busy, ClaudeRunner, Runner

logger = logging.getLogger("desk_bridge")

#: How long one /events poll waits for something to happen.
POLL_SECS = 25.0
#: Events kept for pollers that reconnect.
MAX_EVENTS = 50
#: Longest text sent to the robot in one event; the agent shortens it anyway.
MAX_EVENT_TEXT = 1500
LOCAL_ADDRESSES = ("127.0.0.1", "::1")


class Events:
    """Numbered events with a long-poll wait."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._next_id = 1
        self._changed = asyncio.Condition()

    @property
    def last_id(self) -> int:
        return self._next_id - 1

    async def add(self, kind: str, repo: str, text: str) -> None:
        async with self._changed:
            event = {"id": self._next_id, "kind": kind, "repo": repo, "text": text[-MAX_EVENT_TEXT:]}
            self._next_id += 1
            self._items = [*self._items, event][-MAX_EVENTS:]
            self._changed.notify_all()
        logger.info("event %d %s in %s: %s", event["id"], kind, repo, event["text"][:120])

    async def after(self, last_seen: int, timeout: float) -> list[dict[str, Any]]:
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait_for(lambda: self.last_id > last_seen), timeout)
            except TimeoutError:
                pass
            return [event for event in self._items if event["id"] > last_seen]


class Bridge:
    def __init__(self, config: dict[str, Any]) -> None:
        self.token: str = config["token"]
        self.repos: dict[str, str] = {name: str(Path(path).expanduser()) for name, path in config["repos"].items()}
        self.default_repo: str | None = config.get("default_repo")
        self.announce_after_secs: float = float(config.get("announce_after_secs", 60))
        self.events = Events()
        self.runner: Runner = ClaudeRunner(
            self.events.add, max_budget_usd=float(config.get("max_budget_usd", 5.0)), model=config.get("model")
        )
        #: When each terminal session's current turn started (from the UserPromptSubmit hook).
        self._turn_started: dict[str, float] = {}

    def repo_for(self, name: str | None) -> tuple[str, str]:
        """(name, path) for *name* or the default repo; ValueError with a reason the robot can say."""
        name = (name or self.default_repo or "").strip()
        if not name and len(self.repos) == 1:
            name = next(iter(self.repos))
        if name not in self.repos:
            choices = ", ".join(self.repos) or "none"
            raise ValueError(f"I can only work in these repos: {choices}.")
        return name, self.repos[name]

    def name_of(self, cwd: str) -> str:
        """The allowlisted name of the repo containing *cwd*, or its folder name."""
        for name, path in self.repos.items():
            if cwd == path or cwd.startswith(path.rstrip("/") + "/"):
                return name
        return Path(cwd).name or cwd

    # ---- routes ----

    @web.middleware
    async def auth(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if request.path == "/hook":
            if request.remote not in LOCAL_ADDRESSES:
                return web.json_response({"error": "hooks are only accepted from this machine"}, status=403)
            # A web page open on this laptop could otherwise post fake hook events
            # here. Browsers always send Origin, and a JSON body forces a CORS
            # preflight that this server never answers.
            if "Origin" in request.headers or request.content_type != "application/json":
                return web.json_response({"error": "not a Claude Code hook"}, status=403)
        else:
            sent = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(sent.encode(), self.token.encode()):
                return web.json_response({"error": "wrong or missing token"}, status=401)
        return await handler(request)

    async def get_repos(self, request: web.Request) -> web.Response:
        return web.json_response({"repos": list(self.repos), "default": self.default_repo, "last_event": self.events.last_id})

    async def post_task(self, request: web.Request) -> web.Response:
        body = await request.json()
        task = str(body.get("task") or "").strip()
        if not task:
            return web.json_response({"error": "Say what the task is."}, status=400)
        try:
            name, path = self.repo_for(body.get("repo"))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        try:
            await self.runner.start(task, name, path)
        except Busy:
            return web.json_response({"error": "Another task is still running; wait for it or cancel it."}, status=409)
        return web.json_response(self.runner.status())

    async def get_current(self, request: web.Request) -> web.Response:
        return web.json_response(self.runner.status())

    async def post_message(self, request: web.Request) -> web.Response:
        body = await request.json()
        message = str(body.get("message") or "").strip()
        status = self.runner.status()
        if not message:
            return web.json_response({"error": "Say what to tell Claude."}, status=400)
        if status["state"] == "idle":
            return web.json_response({"error": "There is no task to continue; start one first."}, status=409)
        name, path = self.repo_for(status["repo"])
        try:
            await self.runner.start(message, name, path, follow_up=True)
        except Busy:
            return web.json_response(
                {"error": "Claude is still working on it; wait for it to finish or cancel it."}, status=409
            )
        return web.json_response(self.runner.status())

    async def post_cancel(self, request: web.Request) -> web.Response:
        if not await self.runner.cancel():
            return web.json_response({"error": "Nothing is running."}, status=409)
        return web.json_response({"state": "cancelling"})

    async def get_events(self, request: web.Request) -> web.Response:
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            after = 0
        timeout = min(float(request.query.get("timeout", POLL_SECS)), POLL_SECS)
        return web.json_response({"events": await self.events.after(after, timeout), "last_event": self.events.last_id})

    async def post_hook(self, request: web.Request) -> web.Response:
        """Claude Code hook input (stdin JSON, forwarded by curl). Always 200, so a hook never fails."""
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"ok": False})
        session = body.get("session_id", "")
        hook = body.get("hook_event_name", "")
        if session in self.runner.session_ids:
            return web.json_response({"ok": True, "ignored": "the bridge's own session"})
        repo = self.name_of(body.get("cwd", ""))
        if hook == "UserPromptSubmit":
            self._turn_started[session] = time.monotonic()
        elif hook == "Stop":
            started = self._turn_started.pop(session, None)
            took = time.monotonic() - started if started is not None else 0.0
            if body.get("stop_hook_active"):
                return web.json_response({"ok": True, "ignored": "stop hook already active"})
            if took < self.announce_after_secs:
                return web.json_response({"ok": True, "ignored": f"turn took {took:.0f}s"})
            text = body.get("last_assistant_message") or "It finished its turn."
            await self.events.add("done", repo, text)
        elif hook == "Notification":
            await self.events.add("needs_input", repo, body.get("message") or "Claude Code needs your input.")
        return web.json_response({"ok": True})

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self.auth])
        app.add_routes(
            [
                web.get("/repos", self.get_repos),
                web.post("/tasks", self.post_task),
                web.get("/tasks/current", self.get_current),
                web.post("/tasks/current/messages", self.post_message),
                web.post("/tasks/current/cancel", self.post_cancel),
                web.get("/events", self.get_events),
                web.post("/hook", self.post_hook),
            ]
        )
        return app


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run Claude Code tasks for Palmimo")
    parser.add_argument("--config", type=Path, default=here / "config.toml", help="Path to config.toml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.config.exists():
        raise SystemExit(f"{args.config} not found; copy config.example.toml to config.toml and edit it.")
    config = tomllib.loads(args.config.read_text())
    if config.get("token", "change-me") == "change-me":
        raise SystemExit("Set a real token in config.toml (for example: openssl rand -hex 16).")
    bridge = Bridge(config)
    for name, path in bridge.repos.items():
        if not Path(path).is_dir():
            logger.warning("repo %s: %s does not exist", name, path)
    logger.info("repos: %s; every permission is auto-approved inside them", ", ".join(bridge.repos))
    web.run_app(bridge.app(), host=config.get("host", "0.0.0.0"), port=int(config.get("port", 8765)), print=None)


if __name__ == "__main__":
    main()
