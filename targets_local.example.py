"""
Template for your real monitoring targets.

    cp targets_local.example.py targets_local.py

then fill in TARGETS below with the real, authorized .onion address(es)
you're monitoring. targets_local.py is listed in .gitignore - it will
never be committed or pushed, even if this repo ends up on GitHub
(public or private). monitor.py automatically prefers this file over
its own built-in TARGETS dict if it exists.

Only add addresses you have a legitimate research/OSINT reason to
monitor, and that you've verified against a trusted source - see
tor_monitor_kit/tails_setup_guide.html section 03 for the verification
warning before you paste anything here.
"""

TARGETS = {
    "test_duckduckgo": "http://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
    # "your_site_name": "http://YOUR_REAL_ONION_ADDRESS.onion",
    # "your_site_name_mirror": "http://SUSPECTED_MIRROR_ONION_ADDRESS.onion",
}
