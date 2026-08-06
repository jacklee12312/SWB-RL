# League generation 1 continuation snapshot

This is the latest fully validated and published League generation at the time
of the repository's initial public release. It is a continuation snapshot, not
a claim that generation 1 is the final or strongest possible policy.

The five tar assets contain:

- six active generation-1 lineages at roughly 1.25M agent steps;
- the bounded historical opponent archive and 3M evaluation anchors;
- optimizer, RNG, trainer-counter, version, and experiment-contract state;
- sampler-screen candidates and the frozen training-speed checkpoints;
- payoff evaluations, lineage manifests, queue state, and profiler evidence.

Install every asset with:

```powershell
python -m scripts.bootstrap --release latest
```

The command checks each file against the SHA-256 values committed in
`releases/league-g001-resume.json` before extracting it. See
`docs/CURRENT_RESEARCH_STATE.md` for the exact continuation command and known
limitations.
