from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from swb.rl.tactical_suite import build_tactical_report, load_tactical_case


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Tactical policy evaluation",
        "",
        (
            f"Evaluated {summary['checkpoint_count']} checkpoint(s) on "
            f"{summary['case_count']} case(s): {summary['passed']} passed, "
            f"{summary['failed']} failed."
        ),
        "",
        "| Checkpoint | Case | Top choice | Preferred prob. | Pairwise margin | Result |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in report["results"]:
        target = result["target"]
        lines.append(
            "| {checkpoint} | {case_id} | {choice} | {probability:.6%} | "
            "{margin:+.4f} | {status} |".format(
                checkpoint=Path(result["checkpoint"]["path"]).parent.name,
                case_id=result["case_id"],
                choice=target["selected_action"]["label"],
                probability=target["comparison_preferred_probability"],
                margin=target["pairwise_logit_margin"],
                status="PASS" if target["pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            (
                "`Preferred prob.` is normalized within the annotated preferred "
                "versus disfavored comparison, not across every legal action."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-force tactical replay cases and score PPO checkpoints"
    )
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument(
        "--card-catalog",
        type=Path,
        default=Path("shadowverse_cards.json"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/card_images"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/tactical_suite/evaluation.json"),
    )
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    cases = [load_tactical_case(path) for path in args.case]
    report = build_tactical_report(
        cases,
        args.checkpoint,
        database=args.database,
        card_catalog=args.card_catalog,
        image_directory=args.images,
        device=args.device,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["case_files"] = [path.as_posix() for path in args.case]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
