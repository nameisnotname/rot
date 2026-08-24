"""
generate_test_log.py
====================================================================
Generates a realistic uptime_log.csv for testing analyze_real_log.py's
reporting, WITHOUT waiting for real monitoring hours to pass. Every
derived field (content_simhash, banner detection, content-type guess)
is computed by actually calling signal_utils.py / content_classifier.py
- not hand-typed placeholder values - so the output you see is exactly
what monitor.py would have produced for this content.

ALL DATA THIS SCRIPT GENERATES IS SYNTHETIC - for testing the report
FORMAT only. Never present output from this script as a real monitoring
result. Your actual uptime_log.csv (from monitor.py) is untouched by
this - it writes to a separate test_log_<scenario>.csv file.

USAGE:
    python3 generate_test_log.py [scenario]
    scenario: stable | seizure | ambiguous | ddos   (default: ambiguous)

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

NORMAL_PAGE = "Welcome to Market. Browse categories. csrf_token={t} online_now={n}"
SEIZURE_BANNER = ("This domain has been seized by the Federal Bureau of Investigation "
                   "in accordance with a seizure warrant.")
UNRECOGNIZED_TAKEDOWN = ("Access to this service has been permanently disabled and its "
                          "infrastructure taken into custody following a joint international "
                          "investigation into criminal activity. Evidence has been secured.")


def make_row(t, site_name, status, text=None):
    """Mirrors monitor.py's check_target() field-by-field, minus the
    actual network request - same functions, same logic, so this row
    is indistinguishable from one monitor.py would really write for
    this content."""
    row = {c: "" for c in COLS}
    row.update({
        "timestamp_utc": t.isoformat() + "+00:00",
        "site_name": site_name,
        "url": "http://example.onion",
        "status": status,
    })
    if status == "up":
        row["response_ms"] = 500
        row["http_status"] = 200
        if text is not None:
            row["content_simhash"] = content_simhash(text)
            row["content_length"] = len(text)
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
        row["error"] = "ConnectionError"
        row["error_class"] = "connection_error"
    return row


def gen_stable(hours=60):
    return [make_row(START + timedelta(hours=h), "demo_site", "up",
                      NORMAL_PAGE.format(t=f"tok{h}", n=1000 + h))
            for h in range(hours)]


def gen_seizure(hours=60, outage_at=50):
    rows = []
    for h in range(hours):
        t = START + timedelta(hours=h)
        text = NORMAL_PAGE.format(t=f"tok{h}", n=1000 + h) if h < outage_at else SEIZURE_BANNER
        rows.append(make_row(t, "demo_site", "up", text))
    return rows


def gen_ambiguous(hours=72, drift_at=55, outage_at=60):
    rows = []
    for h in range(hours):
        t = START + timedelta(hours=h)
        if h < drift_at:
            rows.append(make_row(t, "demo_site", "up", NORMAL_PAGE.format(t=f"tok{h}", n=1000 + h)))
        elif h < outage_at:
            rows.append(make_row(t, "demo_site", "up", UNRECOGNIZED_TAKEDOWN))
        else:
            rows.append(make_row(t, "demo_site", "down"))
    return rows


def gen_ddos(hours=60, flap_start=20, flap_end=40):
    rows = []
    up = True
    for h in range(hours):
        t = START + timedelta(hours=h)
        if flap_start <= h < flap_end:
            status = "up" if up else "down"
            up = not up
        else:
            status = "up"
        text = NORMAL_PAGE.format(t=f"tok{h}", n=1000 + h) if status == "up" else None
        rows.append(make_row(t, "demo_site", status, text))
    return rows


SCENARIOS = {"stable": gen_stable, "seizure": gen_seizure, "ambiguous": gen_ambiguous, "ddos": gen_ddos}


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "ambiguous"
    if scenario not in SCENARIOS:
        print(f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}")
        return

    rows = SCENARIOS[scenario]()
    out_path = f"test_log_{scenario}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} SYNTHETIC rows to {out_path} (for testing the report format only)")
    print(f"Now run:  python3 analyze_real_log.py {out_path}")


if __name__ == "__main__":
    main()
