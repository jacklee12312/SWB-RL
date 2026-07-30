# Checklist 1.14 Eight-Deck Gate

- Frozen rules engine: `b6f1d95cd2336cc86772e717e5bd09440a8f38a7`
- Fixed decks: 8
- Direct collectible union / recursive closure: 111 / 147
- Closure collectible / non-collectible: 116 / 31
- Result: 9/9 gates, 147 card rows; PASS

| Gate | Result | Key metrics |
|---|---|---|
| 1.14.1 111-card fixed-deck union and complete recursive closure have final audit rows | passed | fixed_deck_collectible_union_count=111, recursive_reference_count=36, closure_card_count=147, closure_collectible_count=116, closure_non_collectible_count=31 |
| 1.14.2 All applicable alternate modes and cost boundaries pass | passed | training_closure_play_mode_cards=17, play_modes=55, cost_boundary_cases=1546, full_board_cases=55 |
| 1.14.3 All applicable keyword sources and entry methods pass | passed | training_closure_card_count=147, training_keyword_source_count=59, runtime_keyword_count=9, entry_method_count=12 |
| 1.14.4 Target, timing, capacity, and class-resource clauses have direct or generated tests | passed | forced_scenario_assignments=2793, minimum_fixtures_passed=9, minimum_fixture_count=9, direct_state_mutations=17, post_mutation_invariant_checks=35 |
| 1.14.5 Runtime coverage has no unexplained untriggered clause | passed | closure_runtime_clause_count=458, runtime_triggered_passed=15, runtime_explained_by_direct_test=443 |
| 1.14.6 Zero open P0 and P1 bugs | passed | fixed_bug_count=8, total_bug_count=8 |
| 1.14.7 Zero unsupported/placeholder, illegal mutation, and mask mismatch diagnostics | passed | matrix_mask_checks=95230 |
| 1.14.8 At least 1,000 fixed-deck games finish without engine errors and replay by seed | passed | completed_games=1024, random_legal_games=960, frozen_policy_games=64, replay_checks=1024, sampling_strata=128 |
| 1.14.9 Full unit, compile, and required smoke gates pass | passed | unit_tests_passed=2823, unit_tests_conditionally_skipped=1, random_self_play_100_games=100, random_self_play_1000_games=1000, matrix_games=1024 |

## Runtime-coverage interpretation

Raw `not_triggered` clauses are not relabeled as passed. The forced-scenario audit records separate direct-test evidence for every untriggered or unexecuted closure clause.

## Limits

- The 440 not-triggered and 3 triggered-not-executed raw runtime clauses remain honestly labeled; 1.14 acceptance comes from separately re-executed direct tests, not from relabelling sampling coverage.
- This is the eight-fixed-deck gate only. It does not replace the separate 735 collectible + 91 generated full-pool gate.
