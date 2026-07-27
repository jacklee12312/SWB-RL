from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_SCHEMA_VERSION = 2
SUPPORTED_HISTORY_SCHEMA_VERSIONS = frozenset({1, HISTORY_SCHEMA_VERSION})
MATCH_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MatchHistoryStore:
    """Atomic local JSON persistence for simulator match records."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def new_record(
        self,
        *,
        seed: int,
        human_player: int,
        deck: dict[str, Any],
        checkpoint: str,
        warnings: list[str],
        initial_state: dict[str, Any],
        initial_logs: list[str],
    ) -> dict[str, Any]:
        created_at = utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        match_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        record: dict[str, Any] = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "privacy": {
                "persistence": "full",
                "live_ui": "redacted",
            },
            "match_id": match_id,
            "created_at": created_at,
            "updated_at": created_at,
            "seed": seed,
            "human_player": human_player,
            "deck": deck,
            "checkpoint": checkpoint,
            "warnings": list(warnings),
            "status": "ongoing",
            "winner": None,
            "turn": initial_state["turn"],
            "phase": initial_state["phase"],
            "initial_state": initial_state,
            "latest_state": initial_state,
            "logs": list(initial_logs),
            "actions": [],
        }
        self.write(record)
        return record

    def write(self, record: dict[str, Any]) -> None:
        match_id = str(record["match_id"])
        self._validate_match_id(match_id)
        record["updated_at"] = utc_now()
        destination = self.directory / f"{match_id}.json"
        temporary = self.directory / f".{match_id}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, match_id: str) -> dict[str, Any] | None:
        self._validate_match_id(match_id)
        path = self.directory / f"{match_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            not in SUPPORTED_HISTORY_SCHEMA_VERSIONS
            or payload.get("match_id") != match_id
        ):
            raise ValueError(f"invalid match history record: {match_id}")
        return payload

    def list_summaries(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("history limit must be positive")
        summaries: list[dict[str, Any]] = []
        for path in sorted(
            self.directory.glob("*.json"),
            key=lambda candidate: candidate.name,
            reverse=True,
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                match_id = str(payload["match_id"])
                self._validate_match_id(match_id)
                summaries.append(
                    {
                        "match_id": match_id,
                        "created_at": payload["created_at"],
                        "updated_at": payload["updated_at"],
                        "seed": payload["seed"],
                        "status": payload["status"],
                        "winner": payload.get("winner"),
                        "human_player": payload["human_player"],
                        "turn": payload["turn"],
                        "phase": payload["phase"],
                        "action_count": len(payload.get("actions", [])),
                        "deck_display_name": payload["deck"]["display_name"],
                        "checkpoint": payload["checkpoint"],
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                summaries.append(
                    {
                        "match_id": path.stem,
                        "status": "corrupt",
                        "error": str(error),
                    }
                )
            if len(summaries) >= limit:
                break
        return summaries

    @staticmethod
    def _validate_match_id(match_id: str) -> None:
        if not MATCH_ID_PATTERN.fullmatch(match_id):
            raise ValueError("invalid match history id")
