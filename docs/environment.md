# Environment

## Python

- Supported Python: `>=3.11`
- Current verified local version: `Python 3.13.13`
- Current verified pip version: `pip 26.0.1`

## Python Packages

The project currently has no third-party runtime dependencies. The repository
uses only the Python standard library plus local `swb` modules.

`requirements.txt` is intentionally empty except for comments, so standard
setup commands still work:

```powershell
python -m pip install -r requirements.txt
```

## Verification

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```
