"""
confidence_scoring.py
====================================================================
Turns classify()'s per-branch confidence from a hand-picked label
("high if x >= threshold else moderate") into a numeric score derived
from combining the specific evidence relevant to that branch - the
same "combine independent signals, show your work" idea already used
in content_classifier.py's Naive Bayes fusion, applied here to
confidence grading itself.

WHY THIS MATTERS: a fixed threshold can't represent evidence pointing
in OPPOSITE directions at once. The clearest case: heavy target
flapping (attack-like) during a period where a REFERENCE PANEL of
unrelated sites was also degraded (network-event-like, see
METHODOLOGY.md §10). The old design resolved that with a hard
override - the instant the reference-panel check crossed a threshold,
confidence was force-set to "low" regardless of how much other
evidence existed. This module replaces that with genuine fusion.

A DELIBERATE DESIGN CHOICE, caught during calibration testing: for the
ddos_attack case specifically, the two pieces of evidence are combined
MULTIPLICATIVELY in probability space (network-event evidence DISCOUNTS
confidence proportionally), not by subtracting a fixed log-odds
penalty. An earlier version used log-odds subtraction and it produced
a wrong result under testing: with enough transitions, the positive
evidence swamped any fixed penalty, so a target with 50 transitions
during a documented 42%-down reference panel still came out "high"
confidence - which is backwards. A shared network problem produces
extremely high transition counts on its own; heavy local flapping
during a bad network period is NOT independent confirming evidence of
a target-specific attack, it's exactly what a network-wide event looks
like too. A proportional discount (no fixed number of local
transitions can fully rule out a shared cause once a meaningful
fraction of an unrelated panel was also down) is the behavior that's
actually defensible, and it's what's implemented below.

HONEST LIMITATION: the individual log-odds constants below are
HAND-CALIBRATED (chosen so representative feature values land near
sensible confidence-bucket boundaries, matching the old fixed
thresholds' intent), not learned from a validated dataset - same
honesty already applied to signal_utils.py's banner list and
content_classifier.py's training corpus. What's improved here is the
COMBINATION method, not a claim that the individual weights are
empirically optimal - validating those against real data is future
work (METHODOLOGY.md §8), same as elsewhere in this project.
"""

import math


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_to_label(score):
    """Buckets a 0.0-1.0 confidence score into the high/moderate/low
    vocabulary already used throughout this project's output and docs.
    The bucket labels are unchanged - what's different is that they're
    now derived from a real combined score, not the primitive itself."""
    if score is None:
        return "n/a"
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "moderate"
    return "low"


def le_seizure_confidence(content_classifier_corroborates=False):
    """Direct content evidence (an exact banner-phrase match) is the
    strongest evidence type this pipeline can observe - base log-odds
    set high. An independently-computed corroborating signal
    (content_classifier.py separately agreeing) nudges the score
    slightly higher still - though the effect is small and the ceiling
    is capped by the sigmoid, since both signals ultimately read the
    same underlying page content, so they aren't FULLY independent."""
    log_odds = 3.5  # sigmoid(3.5) ~= 0.97
    factors = ["exact banner-phrase match (direct content evidence)"]
    if content_classifier_corroborates:
        log_odds += 0.7
        factors.append("content classifier independently agreed")
    return _sigmoid(log_odds), factors


def migration_confidence(overlap_hrs):
    """Log-odds scales with overlap hours via a log transform
    (diminishing returns - the difference between 2h and 12h of
    confirmed overlap matters far more than 100h vs. 110h). Calibrated
    so ~12h of overlap lands right at the moderate/high boundary,
    matching the old fixed threshold's intent ("high if >= 12h")."""
    log_odds = 0.62 * math.log1p(overlap_hrs) - 0.15
    factors = [f"{overlap_hrs}h of confirmed mirror overlap"]
    return _sigmoid(log_odds), factors


def ddos_confidence(n_transitions, network_event_down_frac=None):
    """Base confidence from transition count alone (log-odds, more
    flapping = stronger evidence). If a reference panel was also
    degraded over the same window, that DISCOUNTS the result
    multiplicatively rather than being subtracted as a fixed penalty -
    see the module docstring for why.

    The discount is QUADRATIC, (1 - frac)^2, not linear - checked
    numerically before shipping: a linear discount only cuts confidence
    by 42% at a 42%-down reference panel, which let a heavily-flapping
    target (e.g. 50 transitions) still land "moderate" even under
    fairly strong shared-cause evidence - looser than the old
    hard-threshold design's intent. A 42%-down reference panel is
    itself fairly strong evidence of a systemic cause (a healthy
    reference site is up the large majority of the time), so the
    discount should grow faster than linearly as that fraction rises.
    Quadratic reproduces the old conservative threshold's behavior at
    its boundary (~30% down -> confidence drops to around the
    moderate/low boundary) while staying continuous instead of a step
    function, and stays gentle at low fractions (10% down barely
    dents a strong base confidence)."""
    log_odds = 0.22 * n_transitions - 0.35  # calibrated: 4 transitions -> moderate, 8 -> high
    factors = [f"{n_transitions} up/down transitions"]
    confidence = _sigmoid(log_odds)
    if network_event_down_frac:
        confidence *= (1 - network_event_down_frac) ** 2
        factors.append(
            f"reference panel also down {network_event_down_frac:.0%} of the same window "
            f"(discounts confidence - consistent with a network-wide event, not necessarily "
            f"a target-specific attack)"
        )
    return confidence, factors


def exit_scam_confidence(deviation_sigmas):
    """Log-odds scales linearly with CUSUM deviation strength (in
    baseline sigmas). Calibrated so the classifier's own decision
    threshold for declaring a changepoint at all (h_sigma=5.0, see
    downtime_cause_classifier.cusum_downshift) sits solidly in
    "moderate", and deviations well past it read "high"."""
    log_odds = 0.42 * deviation_sigmas - 1.75  # calibrated: h_sigma=5.0 (min to detect at all) -> moderate, 8sigma -> high
    factors = [f"CUSUM deviation of {deviation_sigmas:.2f}σ before the outage"]
    return _sigmoid(log_odds), factors


def ambiguous_confidence(content_classifier_hint_confidence=None):
    """The honest fallback - deliberately capped low regardless of any
    single input, since by definition no direct evidence resolved the
    exit-scam-vs-seizure split. A content-classifier hint (unfamiliar
    wording that still resembles a seizure banner) nudges the score up
    slightly - worth a human look - without ever promoting it out of
    the "low" tier: one weak secondary signal shouldn't manufacture
    confidence a genuinely ambiguous case doesn't have."""
    log_odds = -1.5  # sigmoid(-1.5) ~= 0.18
    factors = ["no direct or statistical evidence resolves exit-scam vs. seizure"]
    if content_classifier_hint_confidence:
        log_odds += 0.5 * content_classifier_hint_confidence
        factors.append(
            f"content classifier hint at {content_classifier_hint_confidence:.0%} confidence (unconfirmed)"
        )
    score = min(_sigmoid(log_odds), 0.45)  # hard cap - never leaves the "low" tier
    return score, factors
