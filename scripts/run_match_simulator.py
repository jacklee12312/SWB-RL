from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

from swb.simulator import MatchSimulator
from swb.simulator.server import build_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wait_for_url(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"frontend did not become ready: {url}")


def start_frontend(directory: Path) -> subprocess.Popen:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise FileNotFoundError("npm was not found")
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=directory,
        stdin=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local human-vs-PPO SWB match simulator."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "cards.sqlite3",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "checkpoints"
        / "ppo_evolve_haven_entity_action_100k.pt",
    )
    parser.add_argument(
        "--card-catalog",
        type=Path,
        default=PROJECT_ROOT / "shadowverse_cards.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "card_images",
    )
    parser.add_argument(
        "--history-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "match_history",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-frontend", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    simulator = MatchSimulator(
        database=args.database,
        checkpoint=args.checkpoint,
        card_catalog=args.card_catalog,
        image_directory=args.images,
        history_directory=args.history_directory,
    )
    server = build_server(simulator, host=args.host, port=args.port)
    frontend = None
    try:
        if not args.no_frontend:
            frontend = start_frontend(PROJECT_ROOT / "simulator-ui")
            wait_for_url("http://localhost:3000/")
        if not args.no_browser and not args.no_frontend:
            webbrowser.open("http://localhost:3000/")
        print("SWB simulator ready: http://localhost:3000/", flush=True)
        print(f"Local API: http://{args.host}:{args.port}/api/health")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
            try:
                frontend.wait(timeout=10)
            except subprocess.TimeoutExpired:
                frontend.kill()
                frontend.wait(timeout=5)


if __name__ == "__main__":
    main()
