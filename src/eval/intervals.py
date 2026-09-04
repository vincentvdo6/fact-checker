"""
Confidence intervals on a proportion.

Several numbers in this project are rates over samples small enough that a point estimate invites
over-reading: 101 expert-labelled sentences, a subgroup of 373, a band with 346 members. An
interval is what stops a gap inside the noise from being reported as a finding, so it belongs
somewhere both the evaluation and the audit can reach rather than copied into each.

**Wilson rather than the normal approximation**, because these rates live near the edges. The
normal interval on 0 errors in 40 is [0, 0] -- it claims certainty from the one observation that
provides least -- and above 95% it runs past 1, which reads as a broken number in a report whose
whole argument is carefulness.
"""

from __future__ import annotations

import math


def wilson(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """
    A Wilson score interval, clamped to [0, 1].

    The clamp is not cosmetic. At zero successes the algebra lands a few parts in 10^18 below zero,
    and a printed "-0.0000" in a table of rates is the kind of detail that makes a reader
    reasonably wonder what else was not checked.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))
