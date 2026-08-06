from __future__ import annotations

import argparse
import sys
import unittest
from collections.abc import Iterable


RELEASE_ONLY_MODULES = frozenset(
    {
        "test_ppo_league_baseline",
        "test_ppo_league_evolving",
        "test_ppo_league_generation",
        "test_ppo_league_generation_runner",
        "test_ppo_league_meta_game",
        "test_ppo_league_sampler_screen_results",
        "test_ppo_league_seed_matrix",
        "test_ppo_league_uniform_pool",
        "test_training_speed_stage_2_3",
    }
)


def iter_cases(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_cases(item)
        else:
            yield item


def build_suite(*, without_release: bool) -> tuple[unittest.TestSuite, int]:
    discovered = unittest.defaultTestLoader.discover("tests")
    selected = unittest.TestSuite()
    omitted = 0
    for case in iter_cases(discovered):
        module = case.__class__.__module__.rsplit(".", 1)[-1]
        if without_release and module in RELEASE_ONLY_MODULES:
            omitted += 1
            continue
        selected.addTest(case)
    return selected, omitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the portable CI suite without multi-gigabyte research assets."
    )
    parser.add_argument(
        "--without-release",
        action="store_true",
        help="omit tests whose frozen checkpoints are installed by the research release",
    )
    arguments = parser.parse_args(argv)
    suite, omitted = build_suite(without_release=arguments.without_release)
    if omitted:
        print(
            f"Omitting {omitted} release-only tests; "
            "run scripts.bootstrap --release latest for the complete suite."
        )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
