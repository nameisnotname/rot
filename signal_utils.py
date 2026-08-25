"""
signal_utils.py
====================================================================
Shared, dependency-light signal-extraction helpers used by both the
live Tor monitor (monitor.py) and the offline classifiers
(downtime_cause_classifier.py, analyze_real_log.py).

Deliberately kept free of numpy/pandas/requests so monitor.py (which
runs continuously on a resource-constrained monitoring box) doesn't
have to import the analysis stack, and so analyze_real_log.py doesn't
have to import `requests` just to reuse these functions.
"""

import hashlib
import re

# ---------------------------------------------------------------------------
# Seizure-banner heuristics
# ---------------------------------------------------------------------------
# Publicly documented phrasing patterns seen in known law-enforcement
# takedown banners (e.g. Silk Road, AlphaBay/Hansa "Operation Bayonet",
# Wall Street Market, Hydra, Genesis Market and similar publicized
# seizures - all covered in public court filings / press releases).
# This is a heuristic, NON-EXHAUSTIVE list, not a guarantee of
# detection - extend it as new banner wording is observed in the wild.
# Matching is case-insensitive, over only the first MAX_BANNER_SCAN_BYTES
# of the response body (bounded on purpose - see monitor.py).
#
# Checked against at least one real, documented banner - Nemesis Market's
# actual seizure banner, seized 20 Mar 2024 by Germany's BKA (Frankfurt
# ZIT, with FBI/DEA/IRS-CI involvement), which read "THIS PLATFORM HAS
# BEEN SEIZED ... by the Federal Criminal Police Office in Frankfurt am
# Main". The original pattern list did NOT actually match that text -
# it only covered "domain"/"hidden service"/"website" as the seized
# noun (not "platform") and only "bundeskriminalamt" (the German name)
# rather than BKA's English self-description "Federal Criminal Police
# Office" - caught by testing this claim against the real quote instead
# of trusting the pattern list on sight. Both gaps are fixed below.
# Still a small, curated list, not a measured-precision/recall corpus -
# see METHODOLOGY.md §8/§9 for why that's explicitly future work, not a
# claimed result.

# Bare agency-name patterns (europol|eurojust, department of justice, etc.)
# used to be independent triggers on their own - measured (see
# evaluate_content_signals.py) at only 0.73 precision, because a forum post
# just DISCUSSING another site's seizure ("heard riverside market got seized
# by the fbi") matches identically to a real self-referential banner. Fixed
# by requiring a self-referential deictic phrase (this/the site/domain/
# platform/etc., or the German equivalent) tying the seizure language to
# THIS page specifically, not just proximity to an agency name anywhere in
# the text. Re-measured post-fix at 1.00 precision / 1.00 recall on the
# labeled test set in evaluate_content_signals.py (8 real seizure texts, 13
# negatives including 5 "hard negative" third-party-discussion cases) -
# still a small, hand-curated test set, not a large validated corpus (see
# METHODOLOGY.md §15).
_SELF_REF = (r"(?:this|the|dieser|diese|dieses) (?:hidden service|hidden site|"
             r"domain(?: name)?|website|platform|market(?:place)?|service|site|dienst)")

SEIZURE_BANNER_PATTERNS = [
    rf"{_SELF_REF} (?:has been|was|wurde) (?:permanently )?seized",
    rf"{_SELF_REF} (?:has been|was) (?:taken down|dismantled|shut down)",
    rf"{_SELF_REF}.{{0,80}}seized by (?:the )?(?:fbi|federal bureau of investigation|europol|"
    r"eurojust|homeland security investigations|drug enforcement administration|"
    r"national crime agency|bundeskriminalamt|federal criminal police office)",
    rf"{_SELF_REF}.{{0,80}}(?:beschlagnahmt|im rahmen einer polizeilichen operation)",
    r"in accordance with.{0,40}(?:seizure|forfeiture) (?:order|warrant)",
    r"operation (?:bayonet|disruptor|spectre|sinkhole|panopticon|deep sentinel)",
    r"seizure notice",
]

MAX_BANNER_SCAN_BYTES = 8192


def detect_seizure_banner(text):
    """Scan response text (already truncated by the caller) for known
    seizure-banner phrasing. Returns (matched: bool, pattern: str)."""
    if not text:
        return False, ""
    snippet = text[:MAX_BANNER_SCAN_BYTES].lower()
    for pattern in SEIZURE_BANNER_PATTERNS:
        if re.search(pattern, snippet):
            return True, pattern
    return False, ""


def content_hash(raw_bytes):
    """SHA-256 of the (already bounded) response body. We hash rather
    than store content: enough to detect that a page's content changed
    between checks, without retaining or transmitting scraped
    marketplace/forum content anywhere. NOTE: this is an EXACT hash -
    any single byte of difference (a CSRF token, a nonce, a cache-busting
    query string, a server-rendered "N users online" counter - all common
    on pages with a login/order form) flips it completely, even when
    nothing meaningful changed. Still useful as a cheap "byte-identical
    twice in a row" signal; use content_simhash()/content_similarity()
    below when you actually care about "did this change much."""
    if not raw_bytes:
        return ""
    return hashlib.sha256(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Near-duplicate content fingerprint (SimHash)
# ---------------------------------------------------------------------------
# Standard near-duplicate-detection technique (the same idea search engines
# use to dedupe crawled pages): instead of one exact hash that flips
# completely on any byte difference, build a fixed-size (64-bit) fingerprint
# where each bit is a majority vote over the page's character-shingles, then
# compare two fingerprints by counting differing bits (Hamming distance).
# Small edits (a changed token) flip only a few bits -> high similarity.
# A real content swap (redesign, banner replacing a homepage) flips close to
# half the bits -> low similarity. Storage cost is the same as a hash (16
# hex chars) - no shingle set or content is retained, consistent with the
# content_hash() design above.
SIMHASH_SHINGLE_SIZE = 8  # character n-gram length - see METHODOLOGY.md for the tuning check;
                          # 8 gave clean separation between "one token changed" (~0.85+ similarity)
                          # and "page actually replaced" (~0.4 similarity) on test content; smaller
                          # shingles (e.g. 4) work but with a much thinner margin between the two


def _shingle_hash64(shingle_bytes):
    return int.from_bytes(hashlib.blake2b(shingle_bytes, digest_size=8).digest(), "big")


def content_simhash(text):
    """64-bit near-duplicate fingerprint of `text`, as a 16-char hex
    string. Returns "" for empty input. Whitespace is collapsed and the
    text lowercased first, so incidental formatting differences (not
    real content changes) don't affect the fingerprint."""
    if not text:
        return ""
    normalized = " ".join(text.lower().split())
    if not normalized:
        return ""
    n = SIMHASH_SHINGLE_SIZE
    shingles = [normalized[i:i + n] for i in range(len(normalized) - n + 1)] or [normalized]

    bit_votes = [0] * 64
    for shingle in shingles:
        h = _shingle_hash64(shingle.encode("utf-8"))
        for bit in range(64):
            bit_votes[bit] += 1 if (h >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(64):
        if bit_votes[bit] > 0:
            fingerprint |= (1 << bit)
    return format(fingerprint, "016x")


def content_similarity(fingerprint_hex_a, fingerprint_hex_b):
    """1.0 = identical fingerprints, 0.0 = maximally different (all 64
    bits differ). Returns None if either fingerprint is missing - a
    missing fingerprint (e.g. the check was "down") is "unknown", not
    "different", and callers should treat it that way."""
    if not fingerprint_hex_a or not fingerprint_hex_b:
        return None
    xor = int(fingerprint_hex_a, 16) ^ int(fingerprint_hex_b, 16)
    hamming_distance = bin(xor).count("1")
    return 1 - (hamming_distance / 64)


def classify_request_error(exc):
    """Normalize a requests exception into a coarse failure-mode
    category. Different causes tend to produce different failure
    modes: sustained overload/DDoS -> repeated timeouts; a dead
    relay/circuit -> connection errors; a seized/replaced domain often
    -> no exception at all (a normal response, just different
    content - see detect_seizure_banner above)."""
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout"
    if "SSLError" in name:
        return "tls_error"
    if "ConnectionError" in name:
        return "connection_error"
    return "other"


def classify_http_status(status_code):
    """Coarse error class for a response that came back without an
    exception, based on its HTTP status code."""
    if status_code is None:
        return "other"
    if status_code < 400:
        return "none"
    if status_code < 500:
        return "http_4xx"
    return "http_5xx"


# ---------------------------------------------------------------------------
# HTML-to-visible-text extraction (for content_classifier.py's input only)
# ---------------------------------------------------------------------------
# content_classifier.py was trained on short, clean prose sentences - it has
# never seen a raw HTML document's shape. Feeding it raw markup (tag names,
# attribute values, inline CSS/JS) is a real domain mismatch, not just a
# small-corpus problem: on a real captured page, the word "hidden" showed up
# ~200 times purely from ordinary CSS utility class names
# (hidden-mobile/hidden-desktop/footer-hidden - completely standard
# responsive-design convention), which happens to overlap with the
# classifier's "hidden service" seizure-banner training phrasing and drove a
# false seizure_banner guess at 100% confidence. Reproduced and confirmed
# before writing this fix, not assumed. This function strips tags (and the
# attributes living inside them) plus entire <script>/<style> blocks
# (content included, not just the tags) so the classifier only ever sees
# actual visible page text - matching what it was trained on.
#
# Deliberately NOT used for detect_seizure_banner() or content_simhash():
# the banner-match patterns already tolerate intervening characters
# (`.{0,80}`) and are independently measured at 1.00 precision/recall
# (evaluate_content_signals.py) - changing that input isn't needed. Changing
# content_simhash()'s input would also silently break drift comparisons for
# any monitor.py deployment already mid-collection on the OLD (raw-HTML)
# fingerprint space - a real live-data compatibility risk not worth taking
# for a signal that isn't the one demonstrated to be broken.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html_to_text(html):
    """Visible text only: removes <script>/<style> blocks (tag AND
    content), then strips all remaining tags (which removes their
    attributes too, since an attribute lives inside the tag's own
    angle brackets), then collapses whitespace."""
    if not html:
        return ""
    no_script_style = _SCRIPT_STYLE_RE.sub(" ", html)
    no_tags = _TAG_RE.sub(" ", no_script_style)
    return _WHITESPACE_RE.sub(" ", no_tags).strip()
