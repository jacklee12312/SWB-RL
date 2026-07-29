from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from swb.simulator.service import MatchSimulator


MAX_REQUEST_BYTES = 64 * 1024
LOCAL_FRONTEND_ORIGIN = re.compile(
    r"^http://(?:localhost|127\.0\.0\.1):[0-9]{1,5}$"
)


class SimulatorHTTPServer(ThreadingHTTPServer):
    simulator: MatchSimulator


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    server: SimulatorHTTPServer

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/state":
            try:
                self._send_json(self.server.simulator.state())
            except ValueError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.CONFLICT)
            return
        if path == "/api/history":
            self._send_json(self.server.simulator.list_history())
            return
        if path.startswith("/api/history/"):
            match_id = unquote(path.removeprefix("/api/history/"))
            try:
                self._send_json(
                    self.server.simulator.match_history(match_id)
                )
            except FileNotFoundError as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.NOT_FOUND,
                )
            except ValueError as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return
        if path.startswith("/api/images/"):
            filename = unquote(path.removeprefix("/api/images/"))
            image = self.server.simulator.image_path(filename)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = image.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/new-match":
                state = self.server.simulator.new_match(
                    seed=body.get("seed"),
                    human_player=int(body.get("human_player", 0)),
                    human_deck=body.get("human_deck"),
                    ai_deck=body.get("ai_deck"),
                    model=body.get("model"),
                )
                self._send_json(state)
                return
            if path == "/api/action":
                state = self.server.simulator.apply_human_action(
                    int(body["action"])
                )
                self._send_json(state)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as error:
            self._send_json(
                {"error": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:
            self._send_json(
                {"error": f"simulator error: {error}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if LOCAL_FRONTEND_ORIGIN.fullmatch(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, format: str, *args: object) -> None:
        return


def build_server(
    simulator: MatchSimulator,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> SimulatorHTTPServer:
    server = SimulatorHTTPServer((host, port), SimulatorRequestHandler)
    server.simulator = simulator
    return server
