from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from swb.rl.versioning import stable_json_sha256

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_seed_matrix import (
    OUTPUT_DIRECTORY,
    find_cycles,
)


ROOT = Path(__file__).resolve().parents[1]
PAYOFF_REPORT = OUTPUT_DIRECTORY / "seed_payoff_matrix.json"
DEFAULT_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
SOLVER_TOLERANCE = 1e-10
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPLICATES = 2000
PROFILE_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: str | Path) -> str:
    return _repo_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {_repo_path(path)}")
    return payload


def diagnose_payoff_matrix(
    matrix: Sequence[Sequence[float | None]],
    *,
    tolerance: float = SOLVER_TOLERANCE,
) -> dict[str, object]:
    row_count = len(matrix)
    row_lengths = [len(row) for row in matrix]
    square = bool(row_count) and all(length == row_count for length in row_lengths)
    missing = [
        [row, column]
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value is None
    ]
    non_finite = [
        [row, column]
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value is not None and not math.isfinite(float(value))
    ]
    diagonal_error = None
    antisymmetry_error = None
    rank = None
    all_flat = None
    duplicate_groups: list[list[int]] = []
    if square and not missing and not non_finite:
        array = np.asarray(matrix, dtype=np.float64)
        diagonal_error = float(np.max(np.abs(np.diag(array))))
        antisymmetry_error = float(np.max(np.abs(array + array.T)))
        rank = int(np.linalg.matrix_rank(array, tol=tolerance))
        all_flat = bool(np.max(np.abs(array)) <= tolerance)
        unused = set(range(row_count))
        while unused:
            first = min(unused)
            group = [
                candidate
                for candidate in sorted(unused)
                if np.max(np.abs(array[first] - array[candidate])) <= tolerance
                and np.max(np.abs(array[:, first] - array[:, candidate]))
                <= tolerance
            ]
            for candidate in group:
                unused.remove(candidate)
            if len(group) > 1:
                duplicate_groups.append(group)
    valid = bool(
        square
        and not missing
        and not non_finite
        and diagonal_error is not None
        and diagonal_error <= tolerance
        and antisymmetry_error is not None
        and antisymmetry_error <= tolerance
    )
    warnings = []
    if not square:
        warnings.append("matrix_is_not_square")
    if missing:
        warnings.append("matrix_has_missing_entries")
    if non_finite:
        warnings.append("matrix_has_non_finite_entries")
    if diagonal_error is not None and diagonal_error > tolerance:
        warnings.append("matrix_diagonal_is_not_zero")
    if antisymmetry_error is not None and antisymmetry_error > tolerance:
        warnings.append("matrix_is_not_antisymmetric")
    if rank is not None and rank < row_count:
        warnings.append("matrix_is_singular")
    if duplicate_groups:
        warnings.append("matrix_has_duplicate_strategies")
    if all_flat:
        warnings.append("matrix_is_all_draw")
    return {
        "valid_zero_sum_antisymmetric": valid,
        "shape": [row_count, row_count if square else row_lengths],
        "missing_entries": missing,
        "non_finite_entries": non_finite,
        "max_diagonal_error": diagonal_error,
        "max_antisymmetry_error": antisymmetry_error,
        "rank": rank,
        "duplicate_strategy_groups": duplicate_groups,
        "all_flat": all_flat,
        "warnings": warnings,
        "tolerance": tolerance,
    }


def _solve_square_system(
    coefficients: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    tolerance: float,
) -> list[float] | None:
    size = len(values)
    augmented = [
        [float(value) for value in row] + [float(target)]
        for row, target in zip(coefficients, values)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: (abs(augmented[row][column]), -row),
        )
        if abs(augmented[pivot][column]) <= tolerance:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def solve_zero_sum_meta_strategy(
    matrix: Sequence[Sequence[float | None]],
    *,
    tolerance: float = SOLVER_TOLERANCE,
) -> dict[str, object]:
    diagnostics = diagnose_payoff_matrix(matrix, tolerance=tolerance)
    if not diagnostics["valid_zero_sum_antisymmetric"]:
        raise ValueError(
            "invalid zero-sum payoff matrix: "
            + ",".join(diagnostics["warnings"])
        )
    array = [[float(value) for value in row] for row in matrix]
    size = len(array)
    if diagnostics["all_flat"]:
        weights = [1.0 / size] * size
        return _solver_result(
            array,
            weights,
            diagnostics,
            support=list(range(size)),
            solver_path="all_draw_uniform_canonical_solution",
            tolerance=tolerance,
        )

    candidates: list[dict[str, object]] = []
    for support_size in range(1, size + 1):
        for support in itertools.combinations(range(size), support_size):
            equations = [
                [array[row][column] for row in support]
                for column in support
            ]
            targets = [0.0] * support_size
            equations.append([1.0] * support_size)
            targets.append(1.0)
            for equation_indexes in itertools.combinations(
                range(support_size + 1), support_size
            ):
                solution = _solve_square_system(
                    [equations[index] for index in equation_indexes],
                    [targets[index] for index in equation_indexes],
                    tolerance=tolerance,
                )
                if solution is None or any(value <= tolerance for value in solution):
                    continue
                total = sum(solution)
                solution = [value / total for value in solution]
                equality_residual = max(
                    abs(
                        sum(
                            solution[index] * array[row][column]
                            for index, row in enumerate(support)
                        )
                    )
                    for column in support
                )
                weights = [0.0] * size
                for index, value in zip(support, solution):
                    weights[index] = value
                mixture_vs_pure = [
                    sum(weights[row] * array[row][column] for row in range(size))
                    for column in range(size)
                ]
                worst = min(mixture_vs_pure)
                if equality_residual > tolerance * 100 or worst < -tolerance * 100:
                    continue
                result = _solver_result(
                    array,
                    weights,
                    diagnostics,
                    support=list(support),
                    solver_path="deterministic_support_enumeration_zero_sum_lp_kkt",
                    tolerance=tolerance,
                )
                result["equality_residual"] = equality_residual
                candidates.append(result)
                break

    if not candidates:
        # A skew-symmetric game has value zero, so the row LP reduces to the
        # feasibility polytope p>=0, sum(p)=1, p^T A[:,j]>=0.  Enumerating its
        # vertices covers bootstrap matrices where the canonical row and
        # column equilibrium supports are not identical.
        active_constraints = [
            ("zero_weight", index) for index in range(size)
        ] + [
            ("zero_payoff", index) for index in range(size)
        ]
        for active in itertools.combinations(active_constraints, size - 1):
            equations = [[1.0] * size]
            targets = [1.0]
            for kind, index in active:
                if kind == "zero_weight":
                    equations.append([
                        1.0 if column == index else 0.0
                        for column in range(size)
                    ])
                else:
                    equations.append([
                        array[row][index] for row in range(size)
                    ])
                targets.append(0.0)
            solution = _solve_square_system(
                equations,
                targets,
                tolerance=tolerance,
            )
            if solution is None or any(value < -tolerance * 100 for value in solution):
                continue
            weights = [0.0 if value <= tolerance else value for value in solution]
            total = sum(weights)
            if total <= tolerance:
                continue
            weights = [value / total for value in weights]
            mixture_vs_pure = [
                sum(weights[row] * array[row][column] for row in range(size))
                for column in range(size)
            ]
            if min(mixture_vs_pure) < -tolerance * 100:
                continue
            candidates.append(_solver_result(
                array,
                weights,
                diagnostics,
                support=[
                    index for index, weight in enumerate(weights)
                    if weight > tolerance
                ],
                solver_path="deterministic_zero_sum_lp_vertex_enumeration",
                tolerance=tolerance,
            ))
    if not candidates:
        raise ValueError(
            "zero-sum LP vertex enumeration found no feasible equilibrium; "
            f"diagnostics={diagnostics}"
        )
    candidates.sort(key=lambda result: (
        round(float(result["exploitability_proxy"]), 12),
        -round(float(result["effective_population_size"]), 12),
        -len(result["support"]),
        tuple(result["support"]),
        tuple(round(float(value), 14) for value in result["weights"]),
    ))
    return candidates[0]


def _solver_result(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    diagnostics: Mapping[str, object],
    *,
    support: Sequence[int],
    solver_path: str,
    tolerance: float,
) -> dict[str, object]:
    normalized = [0.0 if abs(value) <= tolerance else float(value) for value in weights]
    total = sum(normalized)
    normalized = [value / total for value in normalized]
    pure_vs_mixture = [
        sum(matrix[row][column] * normalized[column] for column in range(len(matrix)))
        for row in range(len(matrix))
    ]
    mixture_vs_pure = [
        sum(normalized[row] * matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    ]
    worst = min(mixture_vs_pure)
    exploitability = max(0.0, -worst)
    return {
        "weights": normalized,
        "support": list(support),
        "solver_path": solver_path,
        "solver_tolerance": tolerance,
        "mixture_value": sum(
            normalized[row]
            * matrix[row][column]
            * normalized[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        ),
        "mixture_vs_pure_payoffs": mixture_vs_pure,
        "pure_vs_mixture_payoffs": pure_vs_mixture,
        "worst_expected_payoff": worst,
        "internal_best_response_payoff": max(pure_vs_mixture),
        "exploitability_proxy": exploitability,
        "effective_population_size": 1.0 / sum(value * value for value in normalized),
        "diagnostics": dict(diagnostics),
    }


def evaluate_mixture(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> dict[str, object]:
    diagnostics = diagnose_payoff_matrix(matrix)
    return _solver_result(
        matrix,
        weights,
        diagnostics,
        support=[index for index, weight in enumerate(weights) if weight > 0.0],
        solver_path="specified_mixture",
        tolerance=SOLVER_TOLERANCE,
    )


def interquartile_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("IQM requires at least one value")
    ordered = sorted(float(value) for value in values)
    lower = 0.25 * len(ordered)
    upper = 0.75 * len(ordered)
    total = 0.0
    weight = 0.0
    for index, value in enumerate(ordered):
        overlap = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        total += overlap * value
        weight += overlap
    return total / weight


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _candidate_matrix(payoff_report: Mapping[str, object]) -> tuple[list[str], list[list[float]]]:
    candidate_ids = list(payoff_report["models"]["candidate_ids"])
    all_ids = list(payoff_report["payoff_matrix"]["model_ids"])
    source = payoff_report["payoff_matrix"]["antisymmetric_payoff_matrix"]
    indexes = [all_ids.index(model_id) for model_id in candidate_ids]
    return candidate_ids, [
        [float(source[row][column]) for column in indexes]
        for row in indexes
    ]


def _load_candidate_game_tables(
    payoff_report: Mapping[str, object],
    candidate_ids: Sequence[str],
) -> dict[tuple[int, int], dict[tuple[int, int, int, int], float]]:
    index = {model_id: value for value, model_id in enumerate(candidate_ids)}
    tables = {}
    for pair in payoff_report["pairwise_results"]:
        learner = pair["learner_model"]
        opponent = pair["opponent_model"]
        if learner not in index or opponent not in index:
            continue
        raw = _load_json(pair["report"]["path"])
        table = {}
        for game in raw["games"]:
            key = (
                int(game["learner_class_id"]),
                int(game["opponent_class_id"]),
                int(game["deck_index"]),
                int(game["learner_player"]),
            )
            if key in table:
                raise ValueError(f"duplicate paired game key {key}")
            table[key] = float(game["score"])
        if len(table) != 196:
            raise ValueError("candidate pair game table is incomplete")
        tables[(index[learner], index[opponent])] = table
    if len(tables) != 15:
        raise ValueError("expected 15 candidate game tables")
    return tables


def paired_bootstrap(
    payoff_report: Mapping[str, object],
    candidate_ids: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    tables = _load_candidate_game_tables(payoff_report, candidate_ids)
    class_cells = [(left, right) for left in range(1, 8) for right in range(1, 8)]
    rng = random.Random(seed)
    nash_metrics: dict[str, list[float]] = defaultdict(list)
    uniform_metrics: dict[str, list[float]] = defaultdict(list)
    weight_samples = [[] for _ in candidate_ids]
    strategy_payoff_samples = [[] for _ in candidate_ids]
    support_counts = [0] * len(candidate_ids)
    for _ in range(replicates):
        sampled_cells = [class_cells[rng.randrange(len(class_cells))] for _ in class_cells]
        sampled_decks = [
            (rng.randrange(2), rng.randrange(2)) for _ in sampled_cells
        ]
        matrix = [[0.0 for _ in candidate_ids] for _ in candidate_ids]
        for (left, right), table in tables.items():
            scores = []
            for (learner_class, opponent_class), deck_draws in zip(
                sampled_cells, sampled_decks
            ):
                for deck_index in deck_draws:
                    scores.extend([
                        table[(learner_class, opponent_class, deck_index, 0)],
                        table[(learner_class, opponent_class, deck_index, 1)],
                    ])
            payoff = 2.0 * (sum(scores) / len(scores)) - 1.0
            matrix[left][right] = payoff
            matrix[right][left] = -payoff
        nash = solve_zero_sum_meta_strategy(matrix)
        uniform = evaluate_mixture(matrix, [1.0 / len(candidate_ids)] * len(candidate_ids))
        for name in (
            "worst_expected_payoff",
            "exploitability_proxy",
            "effective_population_size",
        ):
            nash_metrics[name].append(float(nash[name]))
            uniform_metrics[name].append(float(uniform[name]))
        for index, weight in enumerate(nash["weights"]):
            weight_samples[index].append(float(weight))
            support_counts[index] += int(weight > SOLVER_TOLERANCE)
        for index, payoff in enumerate(nash["pure_vs_mixture_payoffs"]):
            strategy_payoff_samples[index].append(float(payoff))
    return {
        "method": "paired_hierarchical_class_cell_and_deck_seed_bootstrap",
        "seed": seed,
        "replicates": replicates,
        "resampling_units": {
            "class_cells_per_replicate": 49,
            "deck_match_seeds_per_cell": 2,
            "player_positions_per_deck_seed": 2,
            "common_resample_across_all_model_pairs": True,
        },
        "nash_metric_95_ci": {
            name: _interval(values) for name, values in sorted(nash_metrics.items())
        },
        "uniform_metric_95_ci": {
            name: _interval(values) for name, values in sorted(uniform_metrics.items())
        },
        "nash_weight_95_ci": {
            model_id: _interval(weight_samples[index])
            for index, model_id in enumerate(candidate_ids)
        },
        "nash_support_frequency": {
            model_id: support_counts[index] / replicates
            for index, model_id in enumerate(candidate_ids)
        },
        "strategy_vs_nash_payoff_95_ci": {
            model_id: _interval(strategy_payoff_samples[index])
            for index, model_id in enumerate(candidate_ids)
        },
    }


def _profile_and_iqm(
    matrix: Sequence[Sequence[float]],
    model_ids: Sequence[str],
) -> dict[str, object]:
    score_matrix = [[0.5 * (value + 1.0) for value in row] for row in matrix]
    per_model = {}
    for index, model_id in enumerate(model_ids):
        scores = [
            score
            for opponent, score in enumerate(score_matrix[index])
            if opponent != index
        ]
        per_model[model_id] = {
            "opponent_scores": scores,
            "median": _quantile(scores, 0.5),
            "iqm": interquartile_mean(scores),
            "performance_profile": {
                f"score_at_least_{threshold:.2f}": (
                    sum(score >= threshold for score in scores) / len(scores)
                )
                for threshold in PROFILE_THRESHOLDS
            },
        }
    directed_scores = [
        score_matrix[row][column]
        for row in range(len(model_ids))
        for column in range(len(model_ids))
        if row != column
    ]
    return {
        "per_model": per_model,
        "population_directed_score_iqm": interquartile_mean(directed_scores),
        "population_directed_score_median": _quantile(directed_scores, 0.5),
        "thresholds": list(PROFILE_THRESHOLDS),
    }


def build_meta_game_report(
    *,
    payoff_report_path: str | Path = PAYOFF_REPORT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    payoff_report = _load_json(payoff_report_path)
    model_ids, matrix = _candidate_matrix(payoff_report)
    diagnostics = diagnose_payoff_matrix(matrix)
    nash = solve_zero_sum_meta_strategy(matrix)
    uniform = evaluate_mixture(matrix, [1.0 / len(model_ids)] * len(model_ids))
    bootstrap = paired_bootstrap(
        payoff_report,
        model_ids,
        replicates=bootstrap_replicates,
    )
    for mixture in (nash, uniform):
        mixture["weights_by_model"] = {
            model_id: mixture["weights"][index]
            for index, model_id in enumerate(model_ids)
        }
        mixture["pure_vs_mixture_payoff_by_model"] = {
            model_id: mixture["pure_vs_mixture_payoffs"][index]
            for index, model_id in enumerate(model_ids)
        }
        best_index = max(
            range(len(model_ids)),
            key=lambda index: (mixture["pure_vs_mixture_payoffs"][index], -index),
        )
        mixture["internal_best_response_model"] = model_ids[best_index]

    score_by_edge = {
        (model_ids[row], model_ids[column]): 0.5 * (matrix[row][column] + 1.0)
        for row in range(len(model_ids))
        for column in range(len(model_ids))
        if row != column
    }
    global_cycles = {
        "point_estimate": find_cycles(score_by_edge, model_ids, threshold=0.5),
        "strong_55_percent": find_cycles(score_by_edge, model_ids, threshold=0.55),
    }

    class_cells = payoff_report["aggregates"][
        "candidate_internal_class_cells_both_model_sides"
    ]
    class_ids = [f"class_{class_id}" for class_id in range(1, 8)]
    class_edges = {
        (f"class_{left}", f"class_{right}"): float(
            class_cells[f"{left}_vs_{right}"]["score_rate"]
        )
        for left in range(1, 8)
        for right in range(1, 8)
        if left != right
    }
    class_cycles = {
        "point_estimate": find_cycles(class_edges, class_ids, threshold=0.5),
        "strong_55_percent": find_cycles(class_edges, class_ids, threshold=0.55),
    }

    selection_evidence = []
    for index, model_id in enumerate(model_ids):
        outgoing = [
            opponent
            for opponent in model_ids
            if opponent != model_id and score_by_edge[(model_id, opponent)] > 0.5
        ]
        reasons = ["same_rule_generation_0_candidate"]
        if nash["weights"][index] > SOLVER_TOLERANCE:
            reasons.append("positive_nash_weight")
        if outgoing:
            reasons.append("distinct_point_estimate_winning_edges")
        if bootstrap["nash_support_frequency"][model_id] > 0.0:
            reasons.append("bootstrap_nash_support")
        if len(reasons) == 1:
            reasons.append("uncertainty_retention_no_196_game_elimination")
        selection_evidence.append({
            "model_id": model_id,
            "generation_0_disposition": "include_candidate",
            "nash_weight": nash["weights"][index],
            "bootstrap_support_frequency": bootstrap["nash_support_frequency"][model_id],
            "point_estimate_wins_over": outgoing,
            "reasons": reasons,
        })
    for model_id in payoff_report["models"]["historical_anchor_ids"]:
        selection_evidence.append({
            "model_id": model_id,
            "generation_0_disposition": "include_anchor_only",
            "nash_weight": None,
            "reasons": [
                "historical_anchor",
                "cross_rule_training_history",
                "excluded_from_incomplete_candidate_meta_game",
            ],
        })

    return {
        "schema_version": 1,
        "report_kind": "ppo_league_meta_game",
        "input": {
            "payoff_report": {
                "path": _relative(payoff_report_path),
                "sha256": _sha256_file(payoff_report_path),
                "stable_payload_sha256": stable_json_sha256(payoff_report),
            },
            "candidate_model_ids": model_ids,
            "historical_anchors_excluded_from_meta_game": payoff_report["models"][
                "historical_anchor_ids"
            ],
        },
        "matrix_diagnostics": diagnostics,
        "uniform_mixture": uniform,
        "nash_mixture": nash,
        "bootstrap": bootstrap,
        "robust_aggregates": _profile_and_iqm(matrix, model_ids),
        "cycle_graphs": {
            "global_models": global_cycles,
            "classes": class_cycles,
        },
        "population_selection_evidence": selection_evidence,
        "interpretation_limits": [
            "Exploitability is an internal-population best-response proxy, not a bound against unseen policies.",
            "Bootstrap uncertainty resamples the existing two deck seeds per class cell; it does not replace future 10-seed confirmation.",
            "Historical 3M anchors are not in the Nash solve because anchor-vs-anchor edges were outside the 33-pair contract and their training rules differ.",
        ],
    }


def _percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: Mapping[str, object]) -> bytes:
    nash = report["nash_mixture"]
    uniform = report["uniform_mixture"]
    lines = [
        "# League Meta-game Report",
        "",
        "## Result",
        "",
        "The deterministic zero-sum solve uses only the complete six-model,",
        "same-rule 1M candidate submatrix. Old 3M checkpoints remain anchors.",
        "",
        f"- Uniform worst expected payoff: {uniform['worst_expected_payoff']:+.6f}.",
        f"- Uniform exploitability proxy: {uniform['exploitability_proxy']:.6f}.",
        f"- Nash worst expected payoff: {nash['worst_expected_payoff']:+.6f}.",
        f"- Nash exploitability proxy: {nash['exploitability_proxy']:.6f}.",
        f"- Nash effective population size: {nash['effective_population_size']:.3f}.",
        "",
        "## Mixture weights",
        "",
        "| Model | Uniform | Nash | Nash bootstrap 95% CI | Support frequency | Payoff vs Nash |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    bootstrap = report["bootstrap"]
    for model_id in report["input"]["candidate_model_ids"]:
        interval = bootstrap["nash_weight_95_ci"][model_id]
        lines.append(
            f"| {model_id} | {_percentage(uniform['weights_by_model'][model_id])} | "
            f"{_percentage(nash['weights_by_model'][model_id])} | "
            f"{_percentage(interval[0])}–{_percentage(interval[1])} | "
            f"{_percentage(bootstrap['nash_support_frequency'][model_id])} | "
            f"{nash['pure_vs_mixture_payoff_by_model'][model_id]:+.6f} |"
        )
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- Matrix warnings: {', '.join(report['matrix_diagnostics']['warnings']) or 'none'}.",
        f"- Global point cycles: {len(report['cycle_graphs']['global_models']['point_estimate'])}.",
        f"- Global >55% cycles: {len(report['cycle_graphs']['global_models']['strong_55_percent'])}.",
        f"- Class point cycles: {len(report['cycle_graphs']['classes']['point_estimate'])}.",
        f"- Class >55% cycles: {len(report['cycle_graphs']['classes']['strong_55_percent'])}.",
        "- The exploitability number is only an internal-population proxy; it",
        "  cannot certify robustness against a newly trained best response.",
        "",
    ])
    return "\n".join(lines).encode("utf-8")


def build_outputs(
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, bytes]:
    report = build_meta_game_report(bootstrap_replicates=bootstrap_replicates)
    return {
        "meta_game.json": render_json(report),
        "meta_game.md": render_markdown(report),
    }


def write_outputs(outputs: Mapping[str, bytes], output_directory: str | Path) -> None:
    output = _repo_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output / name).write_bytes(payload)


def check_outputs(outputs: Mapping[str, bytes], output_directory: str | Path) -> list[str]:
    output = _repo_path(output_directory)
    return [
        name
        for name, payload in outputs.items()
        if not (output / name).is_file() or (output / name).read_bytes() != payload
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the League population meta-game.")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    outputs = build_outputs(bootstrap_replicates=args.bootstrap_replicates)
    if args.check:
        mismatches = check_outputs(outputs, args.output_directory)
        if mismatches:
            print("meta-game report mismatch: " + ", ".join(mismatches))
            return 1
        print("meta-game report is byte-stable and current")
        return 0
    write_outputs(outputs, args.output_directory)
    print(f"wrote League meta-game report to {_repo_path(args.output_directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
