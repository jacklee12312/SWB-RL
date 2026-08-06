from __future__ import annotations

import unittest

from scripts.run_ci_tests import RELEASE_ONLY_MODULES, build_suite, iter_cases


class CITestRunnerTests(unittest.TestCase):
    def test_without_release_omits_only_registered_modules(self) -> None:
        suite, omitted = build_suite(without_release=True)
        modules = {
            case.__class__.__module__.rsplit(".", 1)[-1]
            for case in iter_cases(suite)
        }

        self.assertGreater(omitted, 0)
        self.assertTrue(RELEASE_ONLY_MODULES.isdisjoint(modules))
        self.assertIn("test_environment", modules)


if __name__ == "__main__":
    unittest.main()
