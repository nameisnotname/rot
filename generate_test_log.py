"""
generate_test_log.py
====================================================================
Generates a realistic uptime_log.csv for testing analyze_real_log.py's
reporting, WITHOUT waiting for real monitoring hours to pass. Every
derived field (content_simhash, banner detection, content-type guess)
is computed by actually calling signal_utils.py / content_classifier.py
on realistic ~8KB HTML content - not short placeholder sentences - and
truncated the same way monitor.py truncates a real response
(MAX_CONTENT_BYTES = 8192), so content_length, response times, and
classifier behavior all look like genuine captured Tor page fetches,
not obviously synthetic text.

ALL DATA THIS SCRIPT GENERATES IS SYNTHETIC - for testing the report
FORMAT only. Never present output from this script as a real monitoring
result. Your actual uptime_log.csv (from monitor.py) is untouched by
this - it writes to a separate test_log_<scenario>.csv file.

HONEST NOTE ON REALISM: even with full HTML boilerplate, this is still
generated content, not a real capture - a real target's actual markup,
navigation structure, and wording will differ. What this DOES faithfully
reproduce is the shape of a real row: full 8192-byte truncated content,
a real SHA-256/SimHash of that exact content, and the classifier's real
behavior on realistic (not toy) input - including its known false-positive
tendency on generic HTML boilerplate wording, which is exactly what you
may have already seen in your own real log (see chat).

USAGE:
    python3 generate_test_log.py [scenario] [hours]
    scenario: stable | seizure | ambiguous | ddos   (default: ambiguous)
    hours: total hours to generate                  (default: 72)

    stable    - up the whole time, nothing to classify yet
    seizure   - normal, then an exact-match seizure banner -> le_seizure, high confidence
    ambiguous - normal, then UNRECOGNIZED takedown wording, then goes fully dark ->
                the honest "can't tell" verdict, but with a HINT pointing at the
                content classifier's independent suspicion, plus a content-drift event
    ddos      - normal, then a flapping window, then recovers -> ddos_attack

Then analyze it:
    python3 analyze_real_log.py test_log_<scenario>.csv
"""

import csv
import random
import sys
from datetime import datetime, timedelta

from content_classifier import classify_content_type
from signal_utils import content_simhash, detect_seizure_banner

COLS = [
    "timestamp_utc", "site_name", "url", "status", "response_ms", "http_status", "error",
    "error_class", "content_sha256", "content_simhash", "content_length", "content_state",
    "banner_keyword", "server_header", "content_type_guess", "content_type_confidence",
    "content_type_top_words",
]

START = datetime(2026, 8, 1)
MAX_CONTENT_BYTES = 8192  # matches monitor.py's own bound exactly

_MARKET_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<nav><a href="/">Home</a> <a href="/categories">Categories</a> <a href="/support">Support</a></nav>
<main>
<h1>{title}</h1>
<p>{inner}</p>
"""

_MARKET_TAIL = """
</main>
<footer><p>All content copyright the respective site operators.</p></footer>
</body>
</html>
"""

_MARKET_FILLER = ("<p>Browse listings by category, price, and vendor rating. Escrow protects "
                   "both buyers and sellers until delivery is confirmed. Search results update "
                   "as new listings are posted. Contact support for account or order issues.</p>\n")

# A structurally DIFFERENT page shell for notice/takedown content - deliberately not
# a variant of the marketplace template. A real seizure/takedown page is a different
# site entirely (often law-enforcement-hosted), not the same page with one sentence
# swapped - using a shared template for both would make SimHash see them as nearly
# identical (mostly-shared boilerplate), which is NOT what a real page replacement
# looks like and would silently defeat the content-drift detection this scenario is
# meant to demonstrate. Caught by re-running the report after adding realistic HTML
# and seeing content-drift events drop to zero - fixed by giving notice pages their
# own genuinely different structure instead of reusing the marketplace template.
_NOTICE_HEAD = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="text-align:center;background:#000;color:#fff">
<div class="notice-banner">
<h1>{title}</h1>
<p>{inner}</p>
"""

_NOTICE_TAIL = """
</div>
<hr>
<p class="case-ref">Case reference and further information available through official channels.</p>
</body>
</html>
"""

_NOTICE_FILLER = ("<p>This action was taken in accordance with applicable legal authority. "
                   "Any data seized as part of this action may be used as evidence in "
                   "ongoing proceedings. Further inquiries should be directed through "
                   "official channels only.</p>\n")


def _build_page(head, tail, filler, inner, title):
    """Builds a full ~8KB HTML page, then truncates at the exact byte
    boundary monitor.py itself truncates at - so content_length and
    everything derived from it matches a real capture's shape."""
    body = head.format(title=title, inner=inner)
    while len(body.encode("utf-8")) < MAX_CONTENT_BYTES - len(tail.encode("utf-8")):
        body += filler
    body += tail
    raw = body.encode("utf-8")[:MAX_CONTENT_BYTES]
    return raw.decode("utf-8", errors="ignore")


NORMAL_PAGE = lambda h: _build_page(
    _MARKET_HEAD, _MARKET_TAIL, _MARKET_FILLER,
    f"Welcome back. csrf_token=tok{h} online_now={1000 + h}", title="Market")
SEIZURE_PAGE = _build_page(
    _NOTICE_HEAD, _NOTICE_TAIL, _NOTICE_FILLER,
    "This domain has been seized by the Federal Bureau of Investigation "
    "in accordance with a seizure warrant.", title="Notice")
UNRECOGNIZED_TAKEDOWN_PAGE = _build_page(
    _NOTICE_HEAD, _NOTICE_TAIL, _NOTICE_FILLER,
    "Access to this service has been permanently disabled and its infrastructure "
    "taken into custody following a joint international investigation into "
    "criminal activity. Evidence has been secured.", title="Notice")


def make_row(t, site_name, status, text=None):
    """Mirrors monitor.py's check_target() field-by-field, minus the
    actual network request - same functions, same 8192-byte bound, same
    logic - so this row is shaped like one monitor.py would really write
    for this content, not an obviously synthetic placeholder."""
    row = {c: "" for c in COLS}
    row.update({
        "timestamp_utc": t.isoformat() + "+00:00",
        "site_name": site_name,
        "url": "http://exampledemositekjhg7f3.onion",
        "status": status,
    })
    if status == "up":
        row["response_ms"] = random.randint(1100, 3200)  # matches the real spread you're seeing
        row["http_status"] = 200
        row["server_header"] = "nginx"
        if text is not None:
            raw_bytes = text.encode("utf-8")
            row["content_length"] = len(raw_bytes)
            row["content_sha256"] = __import__("hashlib").sha256(raw_bytes).hexdigest()
            row["content_simhash"] = content_simhash(text)
            banner_hit, banner_pattern = detect_seizure_banner(text)
            if banner_hit:
                row["content_state"] = "seizure_banner"
                row["banner_keyword"] = banner_pattern
            else:
                row["content_state"] = "normal"

            ct_label, ct_conf, ct_words = classify_content_type(text)
            row["content_type_guess"] = ct_label or ""
            row["content_type_confidence"] = round(ct_conf, 3) if ct_label else ""
            row["content_type_top_words"] = ",".join(ct_words)
    else:
        row["error"] = "ConnectionTimeout"
        row["error_class"] = "timeout"
    return row


def gen_stable(hours):
    return [make_row(START + timedelta(hours=h), "demo_site", "up", NORMAL_PAGE(h))
            for h in range(hours)]


def gen_seizure(hours):
    outage_at = int(hours * 0.83)  # seizure hits ~5/6 of the way through the window
    rows = []
    for h in range(hours):
        t = START + timedelta(hours=h)
        text = NORMAL_PAGE(h) if h < outage_at else SEIZURE_PAGE
        rows.append(make_row(t, "demo_site", "up", text))
    return rows


def gen_ambiguous(hours):
    drift_at = int(hours * 0.76)
    outage_at = int(hours * 0.83)
    rows = []
    for h in range(hours):
        t = START + timedelta(hours=h)
        if h < drift_at:
            rows.append(make_row(t, "demo_site", "up", NORMAL_PAGE(h)))
        elif h < outage_at:
            rows.append(make_row(t, "demo_site", "up", UNRECOGNIZED_TAKEDOWN_PAGE))
        else:
            rows.append(make_row(t, "demo_site", "down"))
    return rows


def gen_ddos(hours):
    flap_start, flap_end = int(hours * 0.33), int(hours * 0.66)
    rows = []
    up = True
    for h in range(hours):
        t = START + timedelta(hours=h)
        if flap_start <= h < flap_end:
            status = "up" if up else "down"
            up = not up
        else:
            status = "up"
        text = NORMAL_PAGE(h) if status == "up" else None
        rows.append(make_row(t, "demo_site", status, text))
    return rows


SCENARIOS = {"stable": gen_stable, "seizure": gen_seizure, "ambiguous": gen_ambiguous, "ddos": gen_ddos}


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "ambiguous"
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 72
    if scenario not in SCENARIOS:
        print(f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}")
        return

    rows = SCENARIOS[scenario](hours)
    out_path = f"test_log_{scenario}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} SYNTHETIC rows ({hours}h) to {out_path} (for testing the report format only)")
    print(f"Now run:  python3 analyze_real_log.py {out_path}")


if __name__ == "__main__":
    main()
