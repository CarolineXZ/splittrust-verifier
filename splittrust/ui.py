"""Browser UI for the demo: the same scenes, decisions and privacy audit as the CLI, rendered live.

    python -m splittrust.ui --mode live        # then open http://localhost:8765
    python -m splittrust.ui --mode offline     # no API keys needed; scripted actor, heuristic stand-in verifiers

The server wraps one `Demo` instance. Scenes run in a background thread; the page polls /api/events.
Policy edits made in the page are written to policies/private.yaml and the CRM, exactly as the CLI does.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .demo import SCENES, Demo

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
demo: Demo


@app.get("/")
def index():
    return send_file(HERE / "static" / "index.html")


@app.get("/api/state")
def state():
    customers = [{"customer_id": c["id"], "name": c["name"], "email": c["email"],
                  "blocked": any(b["customer_id"] == c["id"] for b in demo.env.blocked_customers())}
                 for c in demo.env.state["db"]["customers"]]
    return jsonify({
        "mode": demo.mode, "models": demo.model_names, "running": demo.running, "scenes": SCENES,
        "counters": demo.counters(), "privacy": demo.privacy(), "customers": customers,
        "policy": {"version": demo.policies.version, "private_yaml": demo.policies.raw_private_yaml(),
                   "private_rendered": demo.policies.render("private", demo.env), "general_rendered": demo.policies.render("general")},
        "n_events": len(demo.events), "replay": demo.replay_all, "scene_order": list(SCENES.keys()),
    })


@app.get("/api/events")
def events():
    since = int(request.args.get("since", 0))
    return jsonify({"events": demo.events[since:], "n_events": len(demo.events), "running": demo.running,
                    "counters": demo.counters(), "privacy": demo.privacy()})


def _start(target, *args) -> tuple:
    if demo.running:
        return jsonify({"error": f"scene {demo.running} is still running"}), 409
    threading.Thread(target=target, args=args, daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/scene/<key>")
def run_scene(key: str):
    if key not in SCENES:
        return jsonify({"error": f"unknown scene {key}"}), 404
    return _start(demo.run_scene, key)


@app.post("/api/task")
def run_task():
    task = (request.json or {}).get("task", "").strip()
    if not task:
        return jsonify({"error": "task is empty"}), 400
    if demo.mode == "offline":
        return jsonify({"error": "custom tasks need --mode live; the offline actor is scripted"}), 400

    def go():
        with demo.lock:
            demo.running = "custom"
            try:
                demo.run_custom(task)
            except Exception as e:  # noqa: BLE001
                demo._emit("error", text=f"task failed: {e}")
            finally:
                demo.running = None

    return _start(go)


@app.post("/api/policy/blocked")
def set_blocked():
    body = request.json or {}
    demo.policies.set_blocked(demo.env, body["customer_id"], bool(body["blocked"]))
    demo.policies.bump_version()
    demo.policies.load()
    who = demo.env.customer_by_id(body["customer_id"])["name"]
    demo._emit("note", text=f"Policy data edited in the UI: {who} {'added to' if body['blocked'] else 'removed from'} the blocked-customer list; "
                            f"policy version {demo.policies.version}.")
    return state()


@app.post("/api/policy/yaml")
def set_yaml():
    text = (request.json or {}).get("text", "")
    try:
        demo.policies.write_private_yaml(text)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"not applied, invalid YAML: {e}"}), 400
    demo._emit("note", text=f"private.yaml edited in the UI; policy version {demo.policies.version}.")
    return state()


@app.post("/api/replay")
def set_replay():
    demo.replay_all = bool((request.json or {}).get("on"))
    return jsonify({"replay": demo.replay_all})


@app.post("/api/reset")
def reset():
    if demo.running:
        return jsonify({"error": f"scene {demo.running} is still running"}), 409
    demo.reset()
    return state()


def main(argv: list[str] | None = None) -> None:
    global demo
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "live"], default="offline")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--replay", action="store_true", help="live verifiers, scripted proposals, for every scene")
    args = ap.parse_args(argv)
    demo = Demo(args.mode, pause=False, replay_all=args.replay, quiet=True)
    print(f"Double Verifier UI · mode={args.mode} · models={demo.model_names}\nopen http://localhost:{args.port}")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    finally:
        demo.restore_private_policy()


if __name__ == "__main__":
    main()