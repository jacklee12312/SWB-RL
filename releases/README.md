# Research snapshots

Large model and league-state snapshots are published as GitHub Release assets,
not committed to Git. Each snapshot contains a manifest, checksums, active
learner checkpoints, every opponent checkpoint required by its published PFSP
lineage manifests, and the research-only sampler/profiler inputs required by
the complete test suite.

Install the latest published snapshot with:

```powershell
python -m scripts.bootstrap --release latest
```

The bootstrap command verifies the asset SHA-256 before extracting it into the
paths expected by the training and evaluation scripts.
