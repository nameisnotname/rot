"""
Real-Log Analyzer
====================================================================
Bridges monitor.py's real uptime log into the SAME feature-extraction
and classification functions used by downtime_cause_classifier.py on
synthetic data - extract_features()/classify() are imported directly,
not reimplemented, so both pipelines are provably running the same
decision logic.

WHAT CHANGED FROM v1: monitor.py used to only record reachability
(up/down). It now also records a content hash, a seizure-banner match,
and a failure-mode class per check (see signal_utils.py). That closes
the original gap documented here: from up/down data ALONE this script
could not tell exit_scam apart from le_seizure - both looked identical
(stable, then one permanent drop). With banner evidence, a real
seizure takedown can now often be identified directly, the same way
the synthetic pipeline does it.

WHAT ALSO CHANGED (a real bug found while testing this against
multiple concurrently-monitored sites): migration detection used to
test every pair of monitored sites for simultaneous uptime, and
flagged a "migration" the instant one site went down while ANY other
monitored site was still up - which is trivially true of any two
unrelated sites you happen to be watching at the same time, not
evidence they're related. Two fixes:
  1. Migration checks are now restricted to pairs YOU explicitly
     declare in KNOWN_MIRROR_PAIRS below - not an automatic fishing
     expedition across every monitored site.
  2. mirror_overlap_hours() (in downtime_cause_classifier.py) now also
     requires the candidate mirror to show a visible recent LAUNCH
     near the original's death, not just coincidental co-uptime.
Sites are also now aligned by their actual UTC timestamps before being
compared (pandas index alignment), not by row position - two sites
whose monitoring started at different times would otherwise get
silently compared hour-N-of-site-A against hour-N-of-site-B instead of
the same real hour.

HONEST REMAINING LIMITATION: not every seizure serves a detectable
banner (some just go dark, some use wording not in the heuristic list
in signal_utils.py), and the CUSUM decline detector needs a real
admin-activity time series (posting frequency, escrow-dispute volume,
etc.) that a passive reachability+content-hash monitor still doesn't
capture. When neither a banner nor a declared mirror match is
observed, this script honestly reports the same
"exit_scam_or_le_seizure (ambiguous)" verdict as before, at low
confidence, rather than guessing.

ALSO NEW: monitor.py now runs a small, from-scratch, interpretable text
classifier (content_classifier.py) on every check - a corroborating
signal, separate from SEIZURE_BANNER_PATTERNS' exact-match list, that
can flag "this looks like a takedown notice" (or error page, or
rebrand notice) based on word-distribution similarity to a curated
corpus, even on wording the exact-match list has never seen. This
script uses it two ways: (1) explaining WHAT KIND of change a
content-drift event was, not just that similarity dropped; (2) as a
corroborating hint in the exit_scam_or_le_seizure ambiguous case - if
the classifier repeatedly leaned "seizure_banner" even though no exact
phrase ever matched, that's surfaced explicitly rather than silently
staying ambiguous. It never overrides the exact-match banner signal,
which stays primary and higher-confidence for the cases it covers.

ALSO NEW: a "ddos_attack" verdict is now cross-checked against a small
REFERENCE_SITE_NAMES panel of known-stable onion mirrors monitored
concurrently. Tor onion services went through a real, sustained
network-wide DDoS period (~June 2022-spring 2023) severe enough to
prompt a proof-of-work defense in Tor 0.4.8 - during a period like
that, flapping shows up on every monitored site, not just the one an
analyst is watching, and a naive classifier would misattribute shared
network conditions to a target-specific attack. If the reference panel
was also significantly down over the same window, the verdict is kept
but downgraded to low confidence with an explicit caution note, rather
than reported as if the attack were confirmed target-specific. See
METHODOLOGY.md §10 for the full writeup.

USAGE:
    python analyze_real_log.py [path/to/uptime_log.csv]
    (defaults to uptime_log.csv produced by monitor.py in this folder)
"""

import os
import sys

import pandas as pd

from confidence_scoring import ambiguous_confidence, ddos_confidence, score_to_label
from downtime_cause_classifier import classify, extract_features, find_final_outage_start
from signal_utils import content_similarity

CONTENT_DRIFT_THRESHOLD = 0.7  # similarity below this, between consecutive checks, is "a real change"

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "uptime_log.csv")

# Explicitly declared candidate mirror/rebrand relationships to check for
# a migration overlap, as (original_site_name, candidate_mirror_site_name)
# pairs using the same site_name keys you set in monitor.py's TARGETS.
# Left empty by default - fill in only pairs you actually suspect are
# related. See the module docstring for why this isn't auto-detected
# across every monitored site.
KNOWN_MIRROR_PAIRS = [
    # ("marketX_main", "marketX_suspected_mirror"),
]

# Known-stable, non-sensitive reference site(s) monitored concurrently with
# real targets, used to tell a target-specific attack apart from a
# network-wide event. Tor onion services suffered a sustained network-wide
# DDoS campaign from roughly June 2022 to spring 2023 (severe enough that
# Tor 0.4.8, released 23 Aug 2023, shipped a proof-of-work defense against
# it) - during a period like that, EVERY onion service flaps, including
# ones nobody is targeting. Without a baseline, that reads as a false
# "ddos_attack" on whatever you happen to be monitoring. The DuckDuckGo
# mirror already monitored as a connectivity smoke test doubles as a free
# first reference site; add other well-known, stable, non-sensitive onion
# mirrors here for a stronger baseline (one is a weak signal, three is
# reasonable).
REFERENCE_SITE_NAMES = ["test_duckduckgo"]
NETWORK_EVENT_OVERLAP_THRESHOLD = 0.3  # reference-panel down-fraction that triggers a caution flag


def load_hourly_status(df, site_name):
    """Collapse raw per-check rows for one site into an hourly up/down
    Series, indexed by real UTC hour. An hour counts as 'up' if any
    check in that hour succeeded. Hours with NO checks at all (monitor
    wasn't running) are forward-filled from the last known status,
    since we have no evidence they changed - flagged in the printed
    report so gaps aren't hidden."""
    site_df = df[df["site_name"] == site_name].copy()
    site_df["timestamp_utc"] = pd.to_datetime(site_df["timestamp_utc"])
    site_df["is_up"] = (site_df["status"] == "up").astype(int)

    hourly = site_df.set_index("timestamp_utc")["is_up"].resample("1h").max()
    n_gaps = int(hourly.isna().sum())
    hourly = hourly.ffill().fillna(0).astype(int)
    return hourly, n_gaps


def load_hourly_banner(df, site_name, index):
    """Collapse per-check content_state into an hourly banner flag: 1
    if ANY check that hour matched a seizure-banner pattern, aligned
    onto `index`. Returns None if the log predates the content-
    forensics columns (older CSV format) so callers can degrade
    gracefully instead of erroring."""
    if "content_state" not in df.columns:
        return None
    site_df = df[df["site_name"] == site_name].copy()
    if site_df.empty:
        return pd.Series(0, index=index)
    site_df["timestamp_utc"] = pd.to_datetime(site_df["timestamp_utc"])
    site_df["is_banner"] = (site_df["content_state"] == "seizure_banner").astype(int)
    hourly = site_df.set_index("timestamp_utc")["is_banner"].resample("1h").max()
    return hourly.reindex(index).fillna(0).astype(int)


def content_drift_summary(df, site_name, threshold=CONTENT_DRIFT_THRESHOLD):
    """Compare each check's content_simhash against the previous check's,
    in chronological order, using the same near-duplicate fingerprint
    monitor.py logs. Returns (avg_similarity, drift_events) where
    drift_events is a list of (timestamp, similarity) pairs for
    consecutive-check similarity drops below `threshold` - i.e. "the
    page changed by more than routine token/counter noise", independent
    of whether it matched a known seizure-banner pattern. Returns
    (None, []) if the log predates content_simhash or there's fewer
    than two comparable checks. A gap (a "down" check with no
    fingerprint) breaks the chain rather than being compared."""
    if "content_simhash" not in df.columns:
        return None, []
    site_df = df[df["site_name"] == site_name].copy()
    site_df["timestamp_utc"] = pd.to_datetime(site_df["timestamp_utc"])
    site_df = site_df.sort_values("timestamp_utc")
    fingerprints = site_df["content_simhash"].fillna("").tolist()
    timestamps = site_df["timestamp_utc"].tolist()

    sims, events = [], []
    for i in range(1, len(fingerprints)):
        sim = content_similarity(fingerprints[i - 1], fingerprints[i])
        if sim is None:
            continue
        sims.append(sim)
        if sim < threshold:
            events.append((timestamps[i], sim))

    avg_sim = sum(sims) / len(sims) if sims else None
    return avg_sim, events


def content_type_lookup(df, site_name):
    """{timestamp -> (label, confidence, top_words_list)} for every check
    of this site with a populated content_classifier.py result. {} if
    the log predates content_type_guess (older CSV format)."""
    if "content_type_guess" not in df.columns:
        return {}
    site_df = df[df["site_name"] == site_name].copy()
    site_df["timestamp_utc"] = pd.to_datetime(site_df["timestamp_utc"])
    lookup = {}
    for _, row in site_df.iterrows():
        label = row.get("content_type_guess", "")
        if not label or pd.isna(label):
            continue
        words = row.get("content_type_top_words", "")
        words_list = words.split(",") if isinstance(words, str) and words else []
        lookup[row["timestamp_utc"]] = (label, row.get("content_type_confidence", ""), words_list)
    return lookup


def seizure_like_hint(df, site_name, min_confidence=0.75):
    """Among checks where the EXACT banner match (content_state) did NOT
    fire, find the strongest check where content_classifier.py still
    guessed seizure_banner at >= min_confidence anyway. Returns
    (timestamp, confidence, top_words) or None. Corroborating-only: this
    never overrides the exact-match signal, which stays primary/
    higher-confidence for the cases it directly covers - it exists for
    the cases the exact-match list misses because the wording is new."""
    if "content_type_guess" not in df.columns:
        return None
    site_df = df[df["site_name"] == site_name].copy()
    if "content_state" in site_df.columns:
        site_df = site_df[site_df["content_state"] != "seizure_banner"]
    conf = pd.to_numeric(site_df["content_type_confidence"], errors="coerce")
    candidates = site_df[(site_df["content_type_guess"] == "seizure_banner") & (conf >= min_confidence)]
    if candidates.empty:
        return None
    best_idx = pd.to_numeric(candidates["content_type_confidence"]).idxmax()
    best = candidates.loc[best_idx]
    words = best["content_type_top_words"].split(",") if isinstance(best["content_type_top_words"], str) else []
    return best["timestamp_utc"], float(best["content_type_confidence"]), words


def declared_mirror_for(name, available_sites):
    for a, b in KNOWN_MIRROR_PAIRS:
        if a == name and b in available_sites:
            return b
        if b == name and a in available_sites:
            return a
    return None


def reference_panel_down_fraction(hourly_by_site, reference_names, index, restrict_to_mask=None):
    """Down-fraction of the declared reference panel, using only
    whichever reference sites have data overlapping `index` (a
    DatetimeIndex). Returns None if no reference site has any
    overlapping data at all - callers must treat that as "unknown",
    not as "network was fine", since silence isn't evidence.

    `restrict_to_mask`, if given, is a boolean array aligned to `index`
    - typically "was the TARGET down this hour." Without it, the
    fraction averages over the target's ENTIRE observed window, which
    is the wrong denominator: a target that flapped hard for a 10-day
    stretch within an otherwise-calm 60-day history gets its true
    correlation diluted by 50 days of irrelevant calm on both sides -
    a real confound during the actual flapping window can end up
    reading as barely-there. Found via the network-DDoS stress test
    (stress_test_ddos_confound.py) producing an implausibly low
    correlation for a scenario deliberately built with perfect
    target/reference correlation during the event window. Restricting
    to exactly the hours the target itself was down answers the
    question that actually matters: "when THIS target went down, was
    the reference panel also down" - not "across this target's whole
    history, how often was the reference panel down.\""""
    fractions = []
    for ref_name in reference_names:
        if ref_name not in hourly_by_site:
            continue
        ref_hourly, _ = hourly_by_site[ref_name]
        overlap_idx = index.intersection(ref_hourly.index)
        if restrict_to_mask is not None:
            masked_idx = index[restrict_to_mask]
            overlap_idx = overlap_idx.intersection(masked_idx)
        if len(overlap_idx) == 0:
            continue
        fractions.append(1 - ref_hourly.reindex(overlap_idx).mean())
    return sum(fractions) / len(fractions) if fractions else None


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    if not os.path.isfile(log_path):
        print(f"No log found at {log_path}. Run monitor.py first to collect real data.")
        return

    df = pd.read_csv(log_path)
    site_names = sorted(df["site_name"].unique())
    has_content_forensics = "content_state" in df.columns

    print("\n" + "=" * 106)
    print("REAL UPTIME LOG ANALYSIS (from monitor.py) - not synthetic")
    if not has_content_forensics:
        print("NOTE: this log predates content-forensics columns (banner detection unavailable for it)")
    print("=" * 106)

    hourly_by_site = {name: load_hourly_status(df, name) for name in site_names}

    for name in site_names:
        hourly, n_gaps = hourly_by_site[name]
        span_hrs = len(hourly)
        if span_hrs < 48:
            print(f"\n{name}: only {span_hrs}h of data collected so far - too little "
                  f"history for a reliable pattern read. Keep monitor.py running longer.")
            continue

        mirror_name = declared_mirror_for(name, hourly_by_site)
        if mirror_name:
            mirror_hourly, _ = hourly_by_site[mirror_name]
            # Align by real UTC hour (index union), not row position -
            # the two sites may not have started being monitored at
            # the same time.
            work_index = hourly.index.union(mirror_hourly.index)
            status_vals = hourly.reindex(work_index, fill_value=0).values
            mirror_vals = mirror_hourly.reindex(work_index, fill_value=0).values
        else:
            work_index = hourly.index
            status_vals = hourly.values
            mirror_vals = None

        banner_hourly = load_hourly_banner(df, name, work_index)
        banner_vals = banner_hourly.values if banner_hourly is not None else None

        # activity=None: a passive reachability+content monitor has no
        # numeric admin-activity proxy (see module docstring) - the
        # shared classifier degrades gracefully instead of guessing.
        feats = extract_features(status_vals, activity=None, mirror_status=mirror_vals, banner=banner_vals)
        pred, confidence, reason, confidence_score = classify(feats)
        if pred == "migration" and mirror_name:
            reason += f" (declared candidate mirror: '{mirror_name}')"

        # Real-data-only evidence the shared, synthetic-compatible classify()
        # doesn't have access to - fused in here via the SAME log-odds/
        # probability functions (confidence_scoring.py), not a hard
        # override, so this is genuine evidence combination end to end
        # rather than shared logic followed by an ad hoc patch.
        if pred == "exit_scam_or_le_seizure":
            hint = seizure_like_hint(df, name)
            if hint is not None:
                ts, hint_conf, words = hint
                confidence_score, _ = ambiguous_confidence(hint_conf)
                confidence = score_to_label(confidence_score)
                reason += (
                    f" - HINT: no exact banner phrase matched, but the content classifier "
                    f"guessed seizure_banner at {hint_conf:.0%} confidence on the check at {ts} "
                    f"(driven by: {', '.join(words)}) - worth a manual look, not confirmed"
                )

        if pred == "ddos_attack":
            ref_names = [r for r in REFERENCE_SITE_NAMES if r != name]
            target_down_mask = (status_vals == 0)
            ref_down_frac = reference_panel_down_fraction(hourly_by_site, ref_names, work_index,
                                                            restrict_to_mask=target_down_mask)
            if ref_down_frac is not None and ref_down_frac >= NETWORK_EVENT_OVERLAP_THRESHOLD:
                confidence_score, _ = ddos_confidence(feats["num_transitions"], ref_down_frac)
                confidence = score_to_label(confidence_score)
                reason += (
                    f" - CAUTION: the reference panel was also down {ref_down_frac:.0%} of this "
                    f"window, consistent with a network-wide event (e.g. a Tor-wide DDoS period) "
                    f"rather than an attack specific to this target; confidence discounted accordingly"
                )

        score_str = f"{confidence_score:.2f}" if confidence_score is not None else "n/a"
        print(f"\n{name}  ({span_hrs}h observed, {n_gaps}h with no check data, forward-filled)")
        print(f"  predicted cause: {pred}  [confidence: {confidence}, score: {score_str}]")
        print(f"  reason: {reason}")

        avg_sim, drift_events = content_drift_summary(df, name)
        if avg_sim is not None:
            print(f"  content stability: {avg_sim:.0%} avg check-to-check similarity "
                  f"({len(drift_events)} significant change event(s) detected)")
            type_lookup = content_type_lookup(df, name)
            for ts, sim in drift_events[-3:]:
                line = f"    - content changed significantly at {ts} (similarity dropped to {sim:.0%})"
                guess = type_lookup.get(ts)
                if guess:
                    label, conf_val, words = guess
                    conf_str = f"{float(conf_val):.0%}" if conf_val != "" else "n/a"
                    line += f" - looks like: {label} ({conf_str}, driven by: {', '.join(words)})"
                print(line)

    print("\n" + "=" * 106 + "\n")


if __name__ == "__main__":
    main()
