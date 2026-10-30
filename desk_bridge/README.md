# Desk bridge

Lets Palmimo run coding tasks with Claude Code on your laptop, and tells Palmimo when they finish. It is a small HTTP server; the robot calls it, never the other way round.

- **By voice:** "Palmimo, ask Claude to add a test for resample." Palmimo starts the task and goes back to sleep. When the task is done, it wakes and tells you the result in one sentence.
- **Your own terminal sessions:** with the hooks below, Palmimo also tells you when a long Claude Code turn finishes or needs your input.

This runs on the laptop only. The Portal never installs it on the robot.

> **Every permission is auto-approved.** Tasks run with `bypassPermissions`: Claude Code edits files and runs commands in the allowlisted repos without asking. Only list repos you are happy for that to happen in, and keep `max_budget_usd` low. Anyone near the robot can start a task.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Claude Code, logged in (`claude` works in a terminal).

```bash
cd desk_bridge
uv sync
cp config.example.toml config.toml
openssl rand -hex 16          # paste as `token` in config.toml
# edit [repos] in config.toml
uv run desk-bridge
```

The bridge listens on port 8765 on every interface, so the robot can reach it on the LAN. Find the laptop's address with `ipconfig getifaddr en0` (macOS).

In the Portal, set these on the app:

| Setting | Value |
| --- | --- |
| `desk_bridge_url` | `http://<laptop-ip>:8765` |
| `DESK_BRIDGE_TOKEN` | the `token` from config.toml |

The robot's log then shows `coding tasks go to http://… (repos: …)`.

Tasks use your normal Claude Code login and settings (CLAUDE.md, MCP servers), and they are billed the same way. A trivial task costs about $0.40 with the default model; set `model` in config.toml to use a cheaper one.

## Hooks for your terminal sessions

Add this to `~/.claude/settings.json`. Merge it into an existing `"hooks"` key if you already have one.

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "curl -s -m 2 -X POST http://127.0.0.1:8765/hook -H 'content-type: application/json' --data-binary @- || true" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "curl -s -m 2 -X POST http://127.0.0.1:8765/hook -H 'content-type: application/json' --data-binary @- || true" }] }],
    "Notification": [{ "hooks": [{ "type": "command", "command": "curl -s -m 2 -X POST http://127.0.0.1:8765/hook -H 'content-type: application/json' --data-binary @- || true" }] }]
  }
}
```

How the bridge handles them:
- A finished turn is announced only if it took at least `announce_after_secs` (60 s by default). Quick back-and-forth in the terminal stays quiet.
- Notifications, such as "Claude needs your permission", are always announced.
- `/hook` only accepts calls from the laptop itself.
- When the bridge isn't running, the hook fails silently and Claude Code carries on.

## API

Every route except `/hook` needs `Authorization: Bearer <token>`. Only one task runs at a time.

| Route | Does |
| --- | --- |
| `GET /repos` | Allowlisted repo names |
| `POST /tasks {task, repo?}` | Start a task (400 for an unknown repo, 409 while another task runs) |
| `GET /tasks/current` | State, latest step, summary, cost |
| `POST /tasks/current/messages {message}` | Follow-up in the same session, once the task has finished |
| `POST /tasks/current/cancel` | Stop the running task |
| `GET /events?after=N` | Long-poll (25 s) for `done`, `failed` and `needs_input` events |
| `POST /hook` | Claude Code hook input |

Try it without the robot:

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST localhost:8765/tasks -d '{"task":"list the TODOs in this repo"}'
curl -H "Authorization: Bearer $TOKEN" localhost:8765/tasks/current
```

## Codex

Not supported yet. `runner.py` has a small `Runner` interface, so a Codex runner (built on `codex exec --json`) can be added beside `ClaudeRunner`.
