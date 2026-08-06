# Security policy

Please report vulnerabilities privately through GitHub's private vulnerability
reporting feature when available. Do not open a public issue containing
credentials, exploitable deployment details, private match records, or other
sensitive data.

The local simulator is a research tool. Do not expose it directly to the
public internet without authentication, request limits, input validation, and
an appropriate production deployment review.

Only load PyTorch checkpoints from trusted sources. Checkpoint files can
contain serialized Python data; verify release checksums before loading them.
