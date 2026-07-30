from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MATRIX = Path(
    "data/reports/card_bug_audit/training_matrix_1000.json"
)
DEFAULT_FULL_POOL = Path(
    "data/reports/card_bug_audit/full_pool_sampling_10000.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "data/reports/card_bug_audit/long_truncation_myuu_distribution.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "data/reports/card_bug_audit/long_truncation_myuu_distribution.md"
)
MYUU_DECK = "international_qr_portal_myuu_20260728"
LONG_AGENT_STEPS = 257


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def _distribution(rows: list[dict[str, object]]) -> dict[str, object]:
    turns = [int(row["turn"]) for row in rows]
    steps = [int(row["agent_steps"]) for row in rows]
    return {
        "games": len(rows),
        "sampling_kinds": dict(sorted(Counter(
            str(row["sampling_kind"]) for row in rows
        ).items())),
        "terminated": sum(bool(row["terminated"]) for row in rows),
        "truncated": sum(bool(row["truncated"]) for row in rows),
        "turn": {
            "min": min(turns),
            "median": _percentile(turns, 0.5),
            "p95": _percentile(turns, 0.95),
            "p99": _percentile(turns, 0.99),
            "max": max(turns),
            "histogram": dict(sorted(
                Counter(str(value) for value in turns).items(),
                key=lambda item: int(item[0]),
            )),
        },
        "agent_steps": {
            "min": min(steps),
            "median": _percentile(steps, 0.5),
            "p95": _percentile(steps, 0.95),
            "p99": _percentile(steps, 0.99),
            "max": max(steps),
            "mean": sum(steps) / len(steps),
        },
    }


def _reproduction_row(
    row: dict[str, object],
    *,
    scope: str,
) -> dict[str, object]:
    keys = (
        "game_id",
        "sampling_kind",
        "class_a",
        "class_b",
        "deck_a_name",
        "deck_b_name",
        "deck_seed_a",
        "deck_seed_b",
        "engine_seed",
        "policy_seed",
        "verify_replay",
        "deck_a_sha256",
        "deck_b_sha256",
        "first_player",
        "winner",
        "turn",
        "agent_steps",
        "terminated",
        "truncated",
        "truncation_reason",
        "mask_mismatches",
        "illegal_actions",
        "placeholder_events",
        "action_trace_sha256",
        "final_fingerprint_sha256",
    )
    return {
        "scope": scope,
        **{key: row[key] for key in keys},
    }


def build_report(
    matrix: dict[str, object],
    full_pool: dict[str, object],
    *,
    matrix_path: Path,
    full_pool_path: Path,
) -> dict[str, object]:
    matrix_rows = list(matrix["games"])
    full_rows = list(full_pool["games"])
    full_p99_turn = int(full_pool["distribution"]["turn_p99"])
    long_turn_threshold = full_p99_turn + 1
    scoped_rows = (
        ("fixed_matrix", matrix_rows),
        ("full_pool", full_rows),
    )
    long_rows = [
        _reproduction_row(row, scope=scope)
        for scope, rows in scoped_rows
        for row in rows
        if (
            int(row["turn"]) >= long_turn_threshold
            or int(row["agent_steps"]) >= LONG_AGENT_STEPS
        )
    ]
    long_rows.sort(
        key=lambda row: (
            -int(row["agent_steps"]),
            -int(row["turn"]),
            str(row["scope"]),
            int(row["game_id"]),
        )
    )
    truncation_rows = [
        _reproduction_row(row, scope=scope)
        for scope, rows in scoped_rows
        for row in rows
        if bool(row["truncated"])
    ]
    myuu_rows = [
        row
        for row in matrix_rows
        if (
            row["deck_a_name"] == MYUU_DECK
            or row["deck_b_name"] == MYUU_DECK
        )
    ]
    myuu_matchups: dict[tuple[str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in myuu_rows:
        side = "player_1" if row["deck_a_name"] == MYUU_DECK else "player_2"
        opponent = (
            str(row["deck_b_name"])
            if side == "player_1"
            else str(row["deck_a_name"])
        )
        myuu_matchups[(opponent, str(row["sampling_kind"]))].append(row)
    myuu_breakdown = []
    for (opponent, sampling_kind), rows in sorted(myuu_matchups.items()):
        myuu_wins = sum(
            (
                row["winner"] == 0
                if row["deck_a_name"] == MYUU_DECK
                else row["winner"] == 1
            )
            for row in rows
        )
        myuu_breakdown.append({
            "opponent": opponent,
            "sampling_kind": sampling_kind,
            "games": len(rows),
            "myuu_wins": myuu_wins,
            "opponent_wins": sum(row["winner"] is not None for row in rows)
            - myuu_wins,
            "draws": sum(row["winner"] is None for row in rows),
            "truncations": sum(bool(row["truncated"]) for row in rows),
            "turn_max": max(int(row["turn"]) for row in rows),
            "agent_steps_max": max(
                int(row["agent_steps"]) for row in rows
            ),
        })
    all_rows = matrix_rows + full_rows
    return {
        "schema_version": 1,
        "report_kind": "swb_card_audit_long_truncation_myuu_distribution",
        "inputs": {
            "fixed_matrix": matrix_path.as_posix(),
            "fixed_matrix_sha256": _sha256(matrix_path),
            "full_pool": full_pool_path.as_posix(),
            "full_pool_sha256": _sha256(full_pool_path),
            "master_seed": full_pool["configuration"]["master_seed"],
            "checkpoint": full_pool["inputs"]["checkpoint"],
            "checkpoint_sha256": full_pool["inputs"]["checkpoint_sha256"],
            "rulebook_sha256": full_pool["inputs"]["rulebook_sha256"],
        },
        "definition": {
            "long_game": (
                f"turn >= {long_turn_threshold} (accepted full-pool p99 "
                f"{full_p99_turn} + 1) or agent_steps >= {LONG_AGENT_STEPS}"
            ),
            "myuu_matchup": (
                f"either fixed-deck side is {MYUU_DECK}"
            ),
            "reproduction_manifest": (
                "scope, full game spec seeds, deck hashes, action trace hash, "
                "terminal fingerprint and outcome"
            ),
        },
        "summary": {
            "source_games": len(all_rows),
            "long_games": len(long_rows),
            "truncations": len(truncation_rows),
            "myuu_games": len(myuu_rows),
            "myuu_truncations": sum(
                bool(row["truncated"]) for row in myuu_rows
            ),
            "passed": (
                bool(matrix["summary"]["passed"])
                and bool(full_pool["summary"]["passed"])
                and not truncation_rows
            ),
        },
        "source_distributions": {
            "fixed_matrix": _distribution(matrix_rows),
            "full_pool": _distribution(full_rows),
        },
        "long_games": {
            "turn_threshold": long_turn_threshold,
            "agent_step_threshold": LONG_AGENT_STEPS,
            "distribution": _distribution([
                row
                for rows in (matrix_rows, full_rows)
                for row in rows
                if (
                    int(row["turn"]) >= long_turn_threshold
                    or int(row["agent_steps"]) >= LONG_AGENT_STEPS
                )
            ]),
            "reproductions": long_rows,
        },
        "truncations": {
            "reproductions": truncation_rows,
            "conclusion": (
                "No truncation reproduction exists because both accepted "
                "source gates recorded zero truncations."
            ),
        },
        "myuu": {
            "deck_name": MYUU_DECK,
            "distribution": _distribution(myuu_rows),
            "matchups": myuu_breakdown,
            "reproductions": [
                _reproduction_row(row, scope="fixed_matrix")
                for row in myuu_rows
            ],
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    full = report["source_distributions"]["full_pool"]
    matrix = report["source_distributions"]["fixed_matrix"]
    myuu = report["myuu"]["distribution"]
    longest = report["long_games"]["reproductions"][:10]
    lines = [
        "# Checklist 1.12 长局、截断与 Myuu 分布",
        "",
        "## 结论",
        "",
        (
            f"- 验收：`{'pass' if summary['passed'] else 'fail'}`；"
            f"来源对局 {summary['source_games']}，长局 {summary['long_games']}，"
            f"截断 {summary['truncations']}，Myuu 对局 {summary['myuu_games']}。"
        ),
        (
            f"- 完整卡池：{full['games']} 局，回合 p99/max "
            f"{full['turn']['p99']}/{full['turn']['max']}，agent steps p99/max "
            f"{full['agent_steps']['p99']}/{full['agent_steps']['max']}。"
        ),
        (
            f"- 八套矩阵：{matrix['games']} 局，回合 p99/max "
            f"{matrix['turn']['p99']}/{matrix['turn']['max']}，agent steps p99/max "
            f"{matrix['agent_steps']['p99']}/{matrix['agent_steps']['max']}。"
        ),
        (
            f"- Myuu：{myuu['games']} 局，截断 {myuu['truncated']}，"
            f"回合 p99/max {myuu['turn']['p99']}/{myuu['turn']['max']}，"
            f"agent steps p99/max "
            f"{myuu['agent_steps']['p99']}/{myuu['agent_steps']['max']}。"
        ),
        "",
        "长局定义："
        f"`{report['definition']['long_game']}`。全部复现 manifest 保存在 JSON。",
        "",
        "## 最长样本",
        "",
        "| scope | game_id | sampling | turn | steps | engine_seed |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            f"| {row['scope']} | {row['game_id']} | "
            f"{row['sampling_kind']} | {row['turn']} | "
            f"{row['agent_steps']} | {row['engine_seed']} |"
        )
        for row in longest
    )
    lines.extend([
        "",
        "## 证据",
        "",
        f"- `{report['inputs']['fixed_matrix']}`",
        f"- `{report['inputs']['full_pool']}`",
        "- JSON 报告保存全部长局、截断和 Myuu 对局的 seed、卡组哈希、"
        "动作轨迹哈希及终局指纹。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report checklist 1.12 long/truncation/Myuu distributions"
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--full-pool", type=Path, default=DEFAULT_FULL_POOL)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    full_pool = json.loads(args.full_pool.read_text(encoding="utf-8"))
    report = build_report(
        matrix,
        full_pool,
        matrix_path=args.matrix,
        full_pool_path=args.full_pool,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        f"acceptance={'pass' if report['summary']['passed'] else 'fail'} "
        f"source_games={report['summary']['source_games']} "
        f"long_games={report['summary']['long_games']} "
        f"truncations={report['summary']['truncations']} "
        f"myuu_games={report['summary']['myuu_games']}"
    )
    if not report["summary"]["passed"]:
        raise SystemExit("long/truncation/Myuu distribution gate failed")


if __name__ == "__main__":
    main()
