"""
Downtime Cause Classifier - Proof of Concept
====================================================================
Poster/paper demo for: "Beyond Up/Down: Classifying the Cause of Dark
Web Marketplace Downtime from Temporal Patterns"

Idea: existing onion-service monitors (TorBot, link directories, etc.)
only report a binary up/down status. This PoC reframes that signal as
a pattern-classification problem: given the *shape* of a site's
uptime/downtime history - plus a couple of passively-observable
content signals - infer the likely CAUSE of an outage: exit scam, law
enforcement seizure, DDoS/attack, or migration.

METHODOLOGY (v2): extract_features()/classify() below are the single
canonical implementation shared by BOTH this synthetic demo and
analyze_real_log.py's real-data pipeline (imported directly, not
reimplemented). Two changes from the original naive version:

  1. Onset-abruptness is no longer a hand-picked fixed slope threshold.
     detect_activity_changepoint() runs a CUSUM control-chart
     changepoint detector (Page, 1954) on the pre-outage activity
     signal, which adapts to that signal's own noise level rather than
     an arbitrary constant. The old fixed-threshold slope check is
     kept alongside it (pre_outage_trend()) purely as a reported
     baseline-comparison metric, not as the decision input.
  2. A seizure-banner signal (see signal_utils.py + monitor.py) can
     override the raw up/down reading: a seizure takedown page often
     still returns HTTP 200, so "reachable" alone would misclassify it
     as up. When banner evidence exists it's treated as the strongest,
     most direct signal available and decided first.

Every prediction also carries a confidence grade (high/moderate/low),
now backed by a real numeric score (see confidence_scoring.py) rather
than a hand-picked per-branch label: each branch's relevant evidence
(banner match strength, CUSUM deviation, transition count, mirror
overlap duration) is combined via log-odds/probability fusion, the
same "combine independent signals, show your work" approach already
used by content_classifier.py's Naive Bayes model. analyze_real_log.py
extends this further for real data by fusing in evidence this
synthetic pipeline doesn't have (a reference-panel network-event
check, a content-classifier corroboration hint) - see METHODOLOGY.md
§13 for the full writeup, including a real calibration bug found and
fixed during testing.

ALL DATA IN THIS SCRIPT IS SYNTHETIC / SIMULATED, generated to match
the qualitative signatures described in the poster abstract. It is
NOT derived from any real onion service, takedown, or observed
marketplace. For demonstration purposes only.

Dependencies: numpy, pandas always (extract_features()/classify()); matplotlib only if you
call plot_timelines() (lazily imported there, not at module level) - analyze_real_log.py
imports this module for extract_features()/classify() alone and never needs matplotlib.
    pip install numpy pandas            # enough for real-data analysis (analyze_real_log.py)
    pip install numpy pandas matplotlib # add this only to regenerate the synthetic figure
"""

import numpy as np
import pandas as pd

from confidence_scoring import (
    ambiguous_confidence,
    ddos_confidence,
    exit_scam_confidence,
    le_seizure_confidence,
    migration_confidence,
    score_to_label,
)

# ---------------------------------------------------------------------------
# STATUS PALETTE (fixed, not themed - from the poster's design system)
# up = "good", down = "critical". Mirror/second domain uses a categorical
# blue so it never gets confused with the up/down status encoding.
# ---------------------------------------------------------------------------
COLOR_UP = "#0ca30c"
COLOR_DOWN = "#d03b3b"
COLOR_MIRROR = "#2a78d6"
COLOR_ACTIVITY = "#4a3aa7"
COLOR_BANNER = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"

RNG_SEED = 42
N_HOURS = 24 * 30  # 30 days, hourly resolution


# ---------------------------------------------------------------------------
# 1. SYNTHETIC TIMELINE GENERATORS
#    Each returns: status (1=up, 0=down), an "admin activity score"
#    (0-100, proxy for posting frequency / withdrawal responsiveness /
#    forum presence), a mirror_status series (None if not applicable),
#    and a banner series (1 = a seizure banner is being served that
#    hour, 0 otherwise) - all hourly over 30 days.
# ---------------------------------------------------------------------------

def gen_exit_scam(rng):
    """Site stays technically reachable while admin activity quietly
    decays for ~6 days, then a single permanent outage with no
    recovery and no seizure banner - it just goes dark."""
    status = np.ones(N_HOURS, dtype=int)
    activity = 75 + rng.normal(0, 4, N_HOURS)

    decline_start, outage_start = 20 * 24, 26 * 24  # day 20 -> day 26
    ramp_len = outage_start - decline_start
    ramp = np.linspace(0, 1, ramp_len)
    activity[decline_start:outage_start] = 75 - ramp * 65 + rng.normal(0, 3, ramp_len)
    activity[outage_start:] = np.clip(5 + rng.normal(0, 2, N_HOURS - outage_start), 0, None)

    status[outage_start:] = 0  # permanent outage, never recovers
    banner = np.zeros(N_HOURS, dtype=int)  # exit scams don't serve a takedown banner
    return status, np.clip(activity, 0, 100), None, banner


def gen_seizure(rng):
    """Site is fully up with normal, stable activity, then vanishes
    instantly and permanently - no decline, no warning. A static
    seizure banner is served from the outage onward, the way real LE
    takedown pages typically behave (often still HTTP 200)."""
    status = np.ones(N_HOURS, dtype=int)
    activity = 78 + rng.normal(0, 4, N_HOURS)  # flat/stable the whole time

    outage_start = 25 * 24
    status[outage_start:] = 0  # instantaneous, no recovery
    activity[outage_start:] = 0

    banner = np.zeros(N_HOURS, dtype=int)
    banner[outage_start:] = 1
    return status, np.clip(activity, 0, 100), None, banner


def gen_ddos(rng):
    """Normal for 5 days, then repeated short up/down flapping for ~10
    days (attack + recovery cycles), then stabilizes back up
    permanently. No banner ever - the service itself never changes."""
    status = np.ones(N_HOURS, dtype=int)
    activity = 72 + rng.normal(0, 5, N_HOURS)

    flap_start, flap_end = 5 * 24, 15 * 24
    hour = flap_start
    up = True
    while hour < flap_end:
        run = rng.integers(1, 6)  # 1-5 hour bursts
        status[hour:min(hour + run, flap_end)] = 1 if up else 0
        if not up:
            activity[hour:min(hour + run, flap_end)] *= 0.15  # activity craters mid-attack
        up = not up
        hour += run
    # day 15 onward: fully recovered and stable, no further outage
    status[flap_end:] = 1
    banner = np.zeros(N_HOURS, dtype=int)
    return status, np.clip(activity, 0, 100), None, banner


def gen_migration(rng):
    """Old domain runs normally, a mirror domain briefly overlaps while
    both are up, then the old domain permanently retires while the
    mirror keeps running. Activity stays stable (an announced move,
    not a quiet disappearance); no seizure banner."""
    status = np.ones(N_HOURS, dtype=int)
    activity = 74 + rng.normal(0, 4, N_HOURS)

    mirror_launch, outage_start = 20 * 24, 22 * 24  # 2-day overlap window
    status[outage_start:] = 0  # old domain permanently retired

    mirror_status = np.zeros(N_HOURS, dtype=int)
    mirror_status[mirror_launch:] = 1  # mirror comes up and stays up

    banner = np.zeros(N_HOURS, dtype=int)
    return status, np.clip(activity, 0, 100), mirror_status, banner


TIMELINES = {
    "MarketA_exit_scam":  ("exit_scam",  gen_exit_scam),
    "ForumB_seizure":     ("le_seizure", gen_seizure),
    "MarketC_ddos":       ("ddos_attack", gen_ddos),
    "MarketD_migration":  ("migration",  gen_migration),
}


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def find_final_outage_start(status):
    """Index where the LAST continuous down-period begins, only if that
    down-period runs uninterrupted through the end of the observation
    window (i.e. a permanent outage with no recovery). Returns None if
    the timeline ends up (site recovered / never had a final outage)."""
    if status[-1] != 0:
        return None
    idx = len(status) - 1
    while idx > 0 and status[idx - 1] == 0:
        idx -= 1
    return idx


def pre_outage_trend(activity, outage_start, window=96, slope_threshold=-0.3):
    """LEGACY / baseline method, kept only for comparison: linear-
    regression slope of activity score in the `window` hours
    immediately before a permanent outage, thresholded at a fixed,
    hand-picked constant. See detect_activity_changepoint() for the
    method actually used by classify() - this one doesn't adapt to the
    signal's noise level, which is exactly the weakness the CUSUM
    version fixes. Returns 'none' if no activity signal is available
    at all (e.g. real ping-only data with no admin-activity proxy)."""
    if activity is None or outage_start is None or outage_start < 10:
        return "none", 0.0
    seg = activity[max(0, outage_start - window):outage_start]
    if len(seg) < 5:
        return "none", 0.0
    slope, _ = np.polyfit(np.arange(len(seg)), seg, 1)
    return ("declining" if slope < slope_threshold else "stable"), slope


def cusum_downshift(series, k_sigma=0.5, h_sigma=5.0, baseline_frac=0.25):
    """Two-sided CUSUM control-chart changepoint detector (Page, 1954)
    for a sustained downward shift in the mean of `series`. Unlike a
    fixed absolute slope threshold, the decision boundary scales with
    the series' own baseline noise (sigma0), so it doesn't need to be
    hand-tuned per deployment.

    k_sigma: allowance/slack before deviations start accumulating,
        in units of baseline sigma (bigger = less sensitive to noise).
    h_sigma: decision threshold - cumulative deviation must exceed
        this many baseline sigmas before a changepoint is declared.
    baseline_frac: fraction of the series (from the start) used to
        estimate the pre-shift baseline mean/sigma.

    Returns (changepoint_index or None, deviation_in_sigmas).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 10:
        return None, 0.0
    baseline = series[: max(5, int(n * baseline_frac))]
    mu0 = baseline.mean()
    sigma0 = baseline.std()
    if sigma0 < 1e-6:
        sigma0 = max(1e-6, abs(mu0) * 0.05)
    k = k_sigma * sigma0
    h = h_sigma * sigma0
    s = 0.0
    for i, x in enumerate(series):
        s = min(0.0, s + (x - mu0) + k)
        if -s > h:
            return i, round(float(-s / sigma0), 2)
    return None, 0.0


def detect_activity_changepoint(activity, outage_start, window=96):
    """Runs CUSUM on the `window` hours immediately before
    `outage_start` to test for a sustained decline (exit-scam-like
    disengagement) vs. a flat baseline (seizure-like, no warning).
    Returns a dict with the verdict, how many hours before the outage
    the shift began, and its strength in baseline sigmas (used for
    confidence grading downstream). Gracefully returns "no evidence"
    when there's no numeric activity signal at all (real ping-only
    data has no admin-activity proxy)."""
    empty = {"declining": False, "changepoint_hr_before_outage": None, "deviation_sigmas": 0.0}
    if activity is None or outage_start is None or outage_start < 10:
        return empty
    seg = activity[max(0, outage_start - window):outage_start]
    cp, dev = cusum_downshift(seg)
    if cp is None:
        return {**empty, "deviation_sigmas": dev}
    return {
        "declining": True,
        "changepoint_hr_before_outage": len(seg) - cp,
        "deviation_sigmas": dev,
    }


def mirror_launch_hour(mirror_status):
    """Index of the first hour a candidate mirror was observed up, or
    None if it was never up - or if it was ALREADY up at the very
    start of the observation window, meaning we have no evidence it's
    a freshly-launched mirror rather than a long-running, unrelated
    site that merely happens to overlap in time."""
    mirror_status = np.asarray(mirror_status)
    if not mirror_status.any():
        return None
    idx = int(np.argmax(mirror_status))
    return None if idx == 0 else idx


def mirror_overlap_hours(status, mirror_status, outage_start, launch_window_hrs=24 * 30):
    """Hours where BOTH the primary and mirror domain were simultaneously
    up, before the primary's permanent outage - counted as migration
    evidence ONLY if the mirror also visibly LAUNCHED (transitioned
    from not-observed-up to up) within launch_window_hrs of that
    outage. Overlap alone is not sufficient: any two unrelated,
    concurrently-monitored sites that both happen to be up will
    "overlap" for as long as both are alive, which would otherwise
    make every co-monitored site look like every other site's mirror
    the moment either one goes down. Requiring a recent, visible
    launch is what actually distinguishes a migration handoff from
    coincidental co-uptime."""
    if mirror_status is None or outage_start is None:
        return 0
    launch_hr = mirror_launch_hour(mirror_status)
    if launch_hr is None or launch_hr > outage_start or (outage_start - launch_hr) > launch_window_hrs:
        return 0
    both_up = (status[:outage_start] == 1) & (mirror_status[:outage_start] == 1)
    return int(both_up.sum())


def extract_features(status, activity, mirror_status, banner=None):
    """Canonical feature extraction, shared by the synthetic demo and
    analyze_real_log.py. `activity` may be None (real ping-only data
    has no admin-activity proxy - decline detection is simply skipped,
    not guessed). `banner` may be None (older logs / no content
    forensics available)."""
    status = np.asarray(status)
    outage_start = find_final_outage_start(status)

    banner_arr = None if banner is None else np.asarray(banner)
    banner_detected = bool(banner_arr is not None and banner_arr.any())
    banner_onset_hr = int(np.argmax(banner_arr)) if banner_detected else None

    # A seizure banner is direct evidence the original service is gone,
    # even on hours where raw HTTP status still reads "up" (a banner
    # page returning 200 OK). The effective outage onset uses whichever
    # evidence - raw unreachability or a banner - comes first.
    effective_outage_start = outage_start
    if banner_detected and (effective_outage_start is None or banner_onset_hr < effective_outage_start):
        effective_outage_start = banner_onset_hr

    trend_label, slope = pre_outage_trend(activity, effective_outage_start)
    cp = detect_activity_changepoint(activity, effective_outage_start)

    overlap_hrs = mirror_overlap_hours(status, mirror_status, effective_outage_start)
    n_transitions = int(np.sum(np.abs(np.diff(status))))

    if effective_outage_start is None:
        abruptness = "n/a (recovered, no permanent outage)"
    elif cp["declining"]:
        abruptness = "gradual"
    else:
        abruptness = "sudden"

    return {
        "final_outage_start_hr": outage_start,
        "effective_outage_start_hr": effective_outage_start,
        "banner_detected": banner_detected,
        "banner_onset_hr": banner_onset_hr,
        "onset_abruptness": abruptness,
        "pre_outage_trend": trend_label,             # legacy fixed-threshold metric (comparison only)
        "trend_slope_per_hr": round(slope, 2),
        "cusum_declining": cp["declining"],           # primary decline signal used by classify()
        "cusum_changepoint_hr_before_outage": cp["changepoint_hr_before_outage"],
        "cusum_deviation_sigmas": cp["deviation_sigmas"],
        "total_downtime_hrs": int(np.sum(status == 0)),
        "num_transitions": n_transitions,
        "mirror_overlap_hrs": overlap_hrs,
        "has_mirror_overlap": overlap_hrs > 0,
    }


# ---------------------------------------------------------------------------
# 3. RULE-BASED CLASSIFIER
#    Deliberately a plain decision path (not a black-box model) so every
#    prediction can be explained in one sentence, with a confidence
#    grade reflecting how strong the corroborating evidence actually
#    is. Mirrors the structure of a shallow decision tree, in the order
#    a human analyst would actually check these signals.
# ---------------------------------------------------------------------------

FLAP_THRESHOLD = 4  # minimum transitions before "recovered" reads as an attack pattern


def classify(f):
    """Returns (predicted_category, confidence_label, reason, confidence_score).

    confidence_score (0.0-1.0 or None for "n/a") comes from
    confidence_scoring.py - a log-odds/probability fusion over the
    evidence relevant to whichever branch fired, replacing the old
    per-branch hand-picked label. confidence_label is that score
    bucketed into the same high/moderate/low vocabulary used
    throughout this project's output, so existing callers/docs that
    only care about the label still work unchanged."""
    if f["banner_detected"]:
        reason = "a known seizure-banner pattern was detected in the response body"
        score, _ = le_seizure_confidence()
        return "le_seizure", score_to_label(score), reason, score

    if f["has_mirror_overlap"]:
        reason = (f"mirror domain overlapped with the original for "
                  f"{f['mirror_overlap_hrs']}h before it went down")
        score, _ = migration_confidence(f["mirror_overlap_hrs"])
        return "migration", score_to_label(score), reason, score

    if f["effective_outage_start_hr"] is None:
        if f["num_transitions"] >= FLAP_THRESHOLD:
            reason = (f"{f['num_transitions']} up/down transitions but no permanent "
                      f"outage in the observation window (site recovered)")
            score, _ = ddos_confidence(f["num_transitions"])
            return "ddos_attack", score_to_label(score), reason, score
        reason = (f"site currently up, only {f['num_transitions']} transition(s) "
                  f"observed - no outage pattern to classify yet")
        return "stable_no_outage", "n/a", reason, None

    if f["cusum_declining"]:
        reason = (f"CUSUM changepoint {f['cusum_changepoint_hr_before_outage']}h before the "
                  f"outage (deviation={f['cusum_deviation_sigmas']}sigma) - admin activity "
                  f"shifted down and stayed down ahead of a permanent, unrecovered outage")
        score, _ = exit_scam_confidence(f["cusum_deviation_sigmas"])
        return "exit_scam", score_to_label(score), reason, score

    reason = ("activity stayed flat/stable right up to an instantaneous, permanent outage, "
              "and no seizure banner was ever observed - the known ambiguous case: could be "
              "a seizure with no banner shown, or an exit scam with an instant rather than "
              "gradual cutoff (needs a signal beyond uptime pings to resolve)")
    score, _ = ambiguous_confidence()
    return "exit_scam_or_le_seizure", score_to_label(score), reason, score


# ---------------------------------------------------------------------------
# 4. RUN CLASSIFIER + PRINT RESULTS TABLE
# ---------------------------------------------------------------------------

def run_and_report():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    raw = {}

    for name, (true_cat, gen_fn) in TIMELINES.items():
        status, activity, mirror, banner = gen_fn(rng)
        feats = extract_features(status, activity, mirror, banner)
        pred_cat, confidence, reason, confidence_score = classify(feats)
        raw[name] = (status, activity, mirror, banner, true_cat, pred_cat, confidence, feats)
        rows.append({
            "timeline": name,
            "true_category": true_cat,
            "predicted": pred_cat,
            "confidence": confidence,
            "confidence_score": round(confidence_score, 3) if confidence_score is not None else None,
            "correct": "YES" if pred_cat == true_cat else "NO",
            "onset": feats["onset_abruptness"],
            "downtime_hrs": feats["total_downtime_hrs"],
            "transitions": feats["num_transitions"],
            "mirror_overlap_hrs": feats["mirror_overlap_hrs"],
            "banner_detected": feats["banner_detected"],
            "cusum_declining": feats["cusum_declining"],
            "key_reason": reason,
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 118)
    print("SYNTHETIC / SIMULATED DATA - FOR DEMONSTRATION PURPOSES ONLY")
    print("Dark Web Marketplace Downtime - Cause Classification PoC (v2: CUSUM + banner-aware)")
    print("=" * 118)

    print("\n--- Extracted features per timeline ---")
    print(df[["timeline", "onset", "downtime_hrs", "transitions",
              "mirror_overlap_hrs", "banner_detected", "cusum_declining"]].to_string(index=False))

    print("\n--- Classification result ---")
    print(df[["timeline", "true_category", "predicted", "confidence", "confidence_score", "correct"]].to_string(index=False))

    print("\n--- Why the classifier decided this (key driving features) ---")
    for _, r in df.iterrows():
        print(f"  {r['timeline']:<20} -> {r['predicted']:<24} [{r['confidence']:<8}] because {r['key_reason']}")

    print("\n" + "=" * 118)
    print("NOTE: all timelines above are synthetically generated for this poster demo.")
    print("They do not represent any real onion service, takedown, or marketplace.")
    print("=" * 118 + "\n")

    return raw


# ---------------------------------------------------------------------------
# 5. PLOT - one small multi-panel figure, up/down step chart per category
# ---------------------------------------------------------------------------

def plot_timelines(raw):
    # Lazy import: matplotlib is only needed for this figure, not for
    # extract_features()/classify() themselves. analyze_real_log.py imports
    # this module for those two functions only and never calls this one -
    # keeping the import here means a Tails install for real-data analysis
    # doesn't need to pull in matplotlib at all (a large, slow package to
    # fetch over Tor for something never used there).
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    hours = np.arange(N_HOURS)
    days = hours / 24

    for ax, (name, (status, activity, mirror, banner, true_cat, pred_cat, confidence, feats)) in zip(
        axes.flat, raw.items()
    ):
        ax.set_facecolor("#fcfcfb")
        ax.fill_between(days, 0, status, step="post", color=COLOR_UP, alpha=0.35, linewidth=0)
        ax.step(days, status, where="post", color=COLOR_UP, linewidth=1.5, label="original domain (up/down)")

        # admin activity score, scaled to [0, 1], overlaid so the pre-outage
        # trend (declining vs stable) is visible - the step chart alone
        # can't distinguish exit-scam from seizure since both end in a
        # single permanent drop.
        ax.plot(days, activity / 100, color=COLOR_ACTIVITY, linewidth=1.2,
                linestyle=":", label="admin activity (scaled)")

        if mirror is not None:
            ax.step(days, mirror * 0.5 + 0.02, where="post", color=COLOR_MIRROR,
                     linewidth=1.5, linestyle="--", label="mirror domain")

        if banner is not None and banner.any():
            ax.fill_between(days, 1.02, 1.10, where=(banner == 1), color=COLOR_BANNER,
                             step="post", linewidth=0, label="seizure banner served")

        ax.set_ylim(-0.1, 1.18)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["down", "up"], color=COLOR_MUTED, fontsize=8)
        ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_MUTED)

        match_mark = "correct" if true_cat == pred_cat else "MISMATCH"
        ax.set_title(
            f"{name}\ntrue: {true_cat}  |  predicted: {pred_cat} [{confidence}]  ({match_mark})",
            fontsize=9, color=COLOR_TEXT, loc="left"
        )
        ax.legend(loc="lower left", fontsize=7, frameon=False)

    for ax in axes[-1, :]:
        ax.set_xlabel("day", color=COLOR_MUTED, fontsize=8)

    fig.suptitle(
        "Synthetic Onion-Service Uptime Patterns by Likely Downtime Cause\n"
        "(SIMULATED DATA - for demonstration purposes only)",
        fontsize=11, color=COLOR_TEXT, y=1.02
    )
    fig.tight_layout()

    out_path = "downtime_patterns.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved figure to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    raw = run_and_report()
    plot_timelines(raw)


if __name__ == "__main__":
    main()
