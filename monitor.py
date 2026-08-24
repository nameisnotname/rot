"""
Onion Service Uptime Monitor
====================================================================
Companion script to downtime_cause_classifier.py.

WHAT THIS DOES: periodically checks whether a given .onion address is
reachable through Tor, and appends the result to a CSV log - not just
up/down anymore, but a small bundle of passively-observable forensic
signals extracted from the SAME request the reachability check already
makes:

  - content_sha256 / content_length: an EXACT hash + size of the
    (bounded) response body - flips completely on any single byte of
    difference, so it only really means "byte-identical to a previous
    check" or not.
  - content_simhash: a 64-bit near-duplicate fingerprint (see
    signal_utils.py) that degrades gracefully instead of flipping
    outright - a changed CSRF token/nonce/live counter (common on any
    page with a login or order form) only shifts it slightly, while an
    actual content swap (redesign, banner replacing the homepage) drops
    it a lot. Compare two fingerprints with signal_utils.content_similarity().
    Neither hash stores or transmits the content itself.
  - content_state / banner_keyword: whether the body matches a known
    law-enforcement seizure-banner pattern. This matters because a
    seizure takedown page often still returns a normal HTTP 200 - a
    naive up/down check would misread a seized site as "up". Matching
    against a small heuristic keyword list (signal_utils.py) fixes
    that without ever browsing, logging in, or scraping beyond the
    root page this monitor already fetches.
  - error_class: normalizes *why* a check failed (timeout vs
    connection error vs TLS error vs HTTP 4xx/5xx) instead of
    collapsing every failure into a bare "down". Different causes tend
    to produce different failure signatures - e.g. sustained overload
    looks like repeated timeouts, a dead circuit looks like connection
    errors.
  - server_header: the `Server` response header, a cheap drift signal
    for "did the backend get redeployed" (e.g. after a migration or a
    takedown-and-replace).
  - content_type_guess / content_type_confidence / content_type_top_words:
    a small, from-scratch, interpretable text classifier (see
    content_classifier.py) that buckets the page into seizure_banner /
    error_maintenance / redesign_rebrand / normal_marketplace based on
    word-distribution similarity to a curated training corpus - a
    corroborating signal that can catch takedown/error/rebrand wording
    SEIZURE_BANNER_PATTERNS' exact-match list has never seen before.
    Every guess logs the specific words that drove it, so it's never a
    bare unexplained label.

That log is the "real data" input downtime_cause_classifier.py's
extract_features()/classify() functions need - the classifier can't
infer anything about a site it has never watched, so this is the
piece that builds up history over days/weeks.

WHAT THIS DELIBERATELY DOES NOT DO: it does not log in, browse
listings, scrape marketplace content, or interact with the site
beyond a single lightweight HTTP request to the root page, and it
reads at most MAX_CONTENT_BYTES of the response body (enough to hash
and pattern-match, not enough to be a scrape). It is a passive
reachability check with light forensics, the same class of check any
uptime monitor (UptimeRobot, TorBot, etc.) performs. Only monitor
sites you have a legitimate research/OSINT reason to observe, and
keep the check interval reasonable (default: hourly) so this never
looks like or behaves like a denial-of-service attempt against the
target.

REQUIREMENTS:
  1. Tor must be running locally with its SOCKS proxy available
     (Tor Browser does this on 127.0.0.1:9150; the standalone `tor`
     daemon / Tails defaults to 127.0.0.1:9050 - adjust
     TOR_SOCKS_PROXY below if needed).
  2. pip install requests pysocks

USAGE:
  1. Fill in TARGETS below with the real .onion addresses you have
     authorization to monitor (left as placeholders here on purpose -
     do not fabricate or guess addresses).
  2. Run a single check:      python monitor.py --once
  3. Or run continuously:     python monitor.py --loop --interval 3600
     (Ctrl+C to stop; safe to re-run --once later via cron / Task
     Scheduler instead of leaving a long-running loop open.)
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import requests

from content_classifier import classify_content_type
from signal_utils import (
    classify_http_status,
    classify_request_error,
    content_hash,
    content_simhash,
    detect_seizure_banner,
)

# ---------------------------------------------------------------------------
# CONFIG - fill these in yourself. Left as placeholders deliberately.
# ---------------------------------------------------------------------------

TARGETS = {
    # friendly_name: onion_url
    #
    # Placeholder target for testing the pipeline end-to-end: DuckDuckGo's
    # official, publicly-published .onion mirror (a legitimate mainstream
    # service's Tor mirror, commonly used just to verify Tor connectivity
    # - not a dark web market). Swap for a real research target once you
    # have one; verify it's still live before relying on it, addresses do
    # change over time.
    "test_duckduckgo": "http://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
    # If you're tracking a suspected mirror/rebrand of the same market,
    # add it here under a second name - analyze_real_log.py looks for
    # exactly this kind of pair to detect a migration overlap window.
    # "site_mirror": "http://REPLACE_WITH_MIRROR_ONION_ADDRESS.onion",
}

# Local, gitignored override: real .onion addresses belong in
# targets_local.py (copy targets_local.example.py to create it), never in
# this file. That way this file - and the whole repo - stays safe to push
# to GitHub (public or private) without ever committing a real target
# address into version control history.
try:
    from targets_local import TARGETS as _LOCAL_TARGETS
    TARGETS = _LOCAL_TARGETS
except ImportError:
    pass

TOR_SOCKS_PROXY = "socks5h://127.0.0.1:9050"  # 9150 if using Tor Browser
PROXIES = {"http": TOR_SOCKS_PROXY, "https": TOR_SOCKS_PROXY}

REQUEST_TIMEOUT_SEC = 20
MAX_CONTENT_BYTES = 8192  # bounded read - enough to hash/pattern-match, not a scrape
LOG_CSV_PATH = os.path.join(os.path.dirname(__file__), "uptime_log.csv")

CSV_COLUMNS = [
    "timestamp_utc", "site_name", "url", "status", "response_ms", "http_status", "error",
    "error_class", "content_sha256", "content_simhash", "content_length", "content_state",
    "banner_keyword", "server_header", "content_type_guess", "content_type_confidence",
    "content_type_top_words",
]


# ---------------------------------------------------------------------------
# CORE CHECK
# ---------------------------------------------------------------------------

def check_target(name, url):
    """One lightweight reachability check + bounded content forensics.
    Returns a log row (dict). No page content is stored - only a hash,
    a length, and a boolean banner match."""
    row = {col: "" for col in CSV_COLUMNS}
    row.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site_name": name,
        "url": url,
        "status": "down",
    })
    start = time.monotonic()
    try:
        resp = requests.get(url, proxies=PROXIES, timeout=REQUEST_TIMEOUT_SEC, stream=True)
        raw_bytes = resp.raw.read(MAX_CONTENT_BYTES, decode_content=True)
        resp.close()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        row["response_ms"] = elapsed_ms
        row["http_status"] = resp.status_code
        row["status"] = "up" if resp.status_code < 500 else "down"
        row["error_class"] = classify_http_status(resp.status_code)
        row["server_header"] = resp.headers.get("Server", "")

        row["content_sha256"] = content_hash(raw_bytes)
        row["content_length"] = len(raw_bytes)
        text = raw_bytes.decode(resp.encoding or "utf-8", errors="ignore")
        row["content_simhash"] = content_simhash(text)
        banner_hit, banner_pattern = detect_seizure_banner(text)
        if banner_hit:
            row["content_state"] = "seizure_banner"
            row["banner_keyword"] = banner_pattern
        elif row["status"] == "up":
            row["content_state"] = "normal"

        # Secondary, corroborating signal (content_classifier.py) - a small
        # trained classifier that can flag e.g. "looks like a takedown
        # notice" even when no exact SEIZURE_BANNER_PATTERNS phrase
        # matched. Logged every check (cheap, pure Python, no dependency),
        # so analyze_real_log.py can use it both to explain content-drift
        # events and as a corroborating hint in the exit-scam/seizure
        # ambiguous case - see METHODOLOGY.md.
        ct_label, ct_confidence, ct_top_words = classify_content_type(text)
        row["content_type_guess"] = ct_label or ""
        row["content_type_confidence"] = round(ct_confidence, 3) if ct_label else ""
        row["content_type_top_words"] = ",".join(ct_top_words)
    except requests.exceptions.RequestException as exc:
        row["error"] = type(exc).__name__
        row["error_class"] = classify_request_error(exc)
    return row


def append_to_log(row, path=LOG_CSV_PATH):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


PLACEHOLDER_MARKERS = ("REPLACE_WITH", "YOUR_REAL_ONION_ADDRESS", "SUSPECTED_MIRROR_ONION_ADDRESS")


def run_once():
    if any(marker in url for url in TARGETS.values() for marker in PLACEHOLDER_MARKERS):
        print("Fill in targets_local.py first: TARGETS still contains placeholder .onion addresses.")
        return

    for name, url in TARGETS.items():
        row = check_target(name, url)
        append_to_log(row)
        flag = f" [{row['content_state'].upper()}]" if row["content_state"] == "seizure_banner" else ""
        print(f"[{row['timestamp_utc']}] {name:<15} -> {row['status']:<5} "
              f"(http={row['http_status'] or '-'}, {row['response_ms'] or '-'}ms, "
              f"{row['error'] or row['error_class'] or 'ok'}){flag}")


def run_loop(interval_sec):
    print(f"Monitoring {len(TARGETS)} target(s) every {interval_sec}s. "
          f"Logging to {LOG_CSV_PATH}. Press Ctrl+C to stop.")
    try:
        while True:
            run_once()
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nStopped.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Passive Tor .onion uptime monitor with content forensics")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--interval", type=int, default=3600,
                         help="seconds between checks in --loop mode (default: 3600 = hourly)")
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()  # default behavior, same as --once


if __name__ == "__main__":
    main()
