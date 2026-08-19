"""Execution verifier for the text->regex SFT task.

The verifier is the ground-truth oracle for the whole pipeline: it decides
which generated patterns are correct, which rejection-sampled traces are kept,
and what the eval metrics are. It is deliberately the only place that executes
model output, so its semantics are defined once.

Three task families:
    validate    -- re.fullmatch must agree with a positive/negative labelling
    extract     -- re.search group(1) must equal an expected substring
    substitute  -- re.sub with a replacement must produce an expected string

Note on `extract`/`validate` scanning: we use finditer + group(0) rather than
findall, because findall silently switches from returning whole matches to
returning group tuples as soon as the pattern contains a capture group. That
would make the same verification code report spurious failures on multi-turn
trajectories where a later turn introduces groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Whitelisted so specs stay reproducible and models cannot smuggle in
# behaviour-changing flags (e.g. re.VERBOSE silently ignoring whitespace).
FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}

# Guards against catastrophic backtracking taking down the training loop.
MAX_PATTERN_LEN = 400


class VerifierError(Exception):
    """Raised when a candidate cannot be evaluated at all (vs. merely wrong)."""


@dataclass
class Result:
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def compiled(self) -> bool:
        return self.error is None


def compile_pattern(pattern: str, flags: str = "") -> re.Pattern:
    if not isinstance(pattern, str):
        raise VerifierError(f"pattern must be a string, got {type(pattern).__name__}")
    if len(pattern) > MAX_PATTERN_LEN:
        raise VerifierError(f"pattern exceeds {MAX_PATTERN_LEN} chars")
    bits = 0
    for ch in flags or "":
        if ch not in FLAG_MAP:
            raise VerifierError(f"unsupported flag {ch!r}; allowed: {sorted(FLAG_MAP)}")
        bits |= FLAG_MAP[ch]
    try:
        return re.compile(pattern, bits)
    except re.error as exc:
        raise VerifierError(f"compile error: {exc}") from exc


def _verify_validate(rx: re.Pattern, cases: dict) -> list[str]:
    failures = []
    for s in cases.get("positives", []):
        if not rx.fullmatch(s):
            failures.append(f"should match but did not: {s!r}")
    for s in cases.get("negatives", []):
        if rx.fullmatch(s):
            failures.append(f"should NOT match but did: {s!r}")
    return failures


def _verify_extract(rx: re.Pattern, cases: list) -> list[str]:
    failures = []
    for text, expected in cases:
        m = rx.search(text)
        if expected is None:
            if m:
                failures.append(f"expected no match on {text!r}, got {m.group(0)!r}")
            continue
        if not m:
            failures.append(f"no match on {text!r}, expected capture {expected!r}")
            continue
        if m.re.groups < 1:
            failures.append(f"pattern has no capture group (needed for {text!r})")
            continue
        if m.group(1) != expected:
            failures.append(f"on {text!r} captured {m.group(1)!r}, expected {expected!r}")
    return failures


def _verify_substitute(rx: re.Pattern, replacement: str, cases: list) -> list[str]:
    failures = []
    for text, expected in cases:
        try:
            got = rx.sub(replacement, text)
        except re.error as exc:
            # A bad group reference in the replacement only surfaces here.
            failures.append(f"substitution error on {text!r}: {exc}")
            continue
        if got != expected:
            failures.append(f"on {text!r} produced {got!r}, expected {expected!r}")
    return failures


def verify(candidate: dict, spec: dict, split: str = "hidden") -> Result:
    """Check one candidate {pattern, flags, replacement?} against a spec split.

    Returns a Result rather than raising for wrong-but-runnable answers, so the
    caller can distinguish "model produced an invalid regex" from "model
    produced a valid regex that is incorrect" -- those are different errors and
    we report them separately in the eval.
    """
    family = spec["family"]
    cases = spec[split]

    try:
        rx = compile_pattern(candidate.get("pattern", ""), candidate.get("flags", ""))
    except VerifierError as exc:
        return Result(passed=False, error=str(exc))

    if family == "validate":
        failures = _verify_validate(rx, cases)
    elif family == "extract":
        failures = _verify_extract(rx, cases)
    elif family == "substitute":
        replacement = candidate.get("replacement")
        if replacement is None:
            return Result(passed=False, error="substitute task requires a 'replacement'")
        failures = _verify_substitute(rx, replacement, cases)
    else:
        raise VerifierError(f"unknown family {family!r}")

    return Result(passed=not failures, failures=failures)


def verify_both_splits(candidate: dict, spec: dict) -> dict:
    """Score on visible and hidden. The visible-pass/hidden-fail combination is
    our overfitting signal: the candidate reproduced the shown examples without
    honouring the instruction."""
    vis = verify(candidate, spec, "visible")
    hid = verify(candidate, spec, "hidden")
    return {
        "visible_pass": vis.passed,
        "hidden_pass": hid.passed,
        "compiled": vis.compiled,
        "overfit": vis.passed and not hid.passed,
        "error": vis.error,
        "failures": hid.failures[:5],
    }
