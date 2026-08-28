"""
content_classifier.py
====================================================================
A small, from-scratch, interpretable text classifier (multinomial Naive
Bayes, trained on a curated corpus baked into this file) that buckets a
fetched page's content into one of four categories: seizure_banner,
error_maintenance, redesign_rebrand, normal_marketplace - or, when the
page's real content doesn't actually give it enough to go on,
"uncertain" rather than a confident wrong guess (see "WHY 'uncertain'
EXISTS" below).

WHY THIS EXISTS: SEIZURE_BANNER_PATTERNS in signal_utils.py is an exact
regex/keyword match - it only catches wording it has seen before. This
classifier is a genuinely different, complementary signal: it scores
NEW/unseen wording by how similar its word distribution is to each
known category, so it can flag "this looks like a takedown notice" even
when no exact phrase matches - and it distinguishes an error/maintenance
page from a redesign notice from a takedown notice, three things the
exact-match banner list conflates into a single "not normal" bucket.

WHY NAIVE BAYES, FROM SCRATCH, NO DEPENDENCIES (not an embedding/deep
model): consistent with the rest of this project's design philosophy
(see info.html's Method section on why classify() itself is rule-based,
not learned) -
  1. Pure Python/stdlib only - installs instantly on Tails, no large
     model-weight download over Tor, no scikit-learn/numpy/torch
     dependency chain to fight with on a resource-constrained box.
  2. A handful of hand-written training examples per category is a
     small-data regime where a bigger model would just memorize, not
     generalize - the exact same reasoning already applied to why
     classify() is a rule-based decision path rather than a trained
     tree.
  3. Every prediction can show its work (the words that most drove the
     decision), so a result is auditable, not a black box - matching
     this project's core design value across every other module.

REAL BUG FOUND AND FIXED HERE (2026-08-28, from a manual ground-truth
spot check - see quickstart_kit/readthis.md and Book1-tor.csv): two
LIVE, normal marketplace pages (a WooCommerce carding shop and a
Bitcoin/Monero-only shop's homepage) were confidently misclassified as
seizure_banner at 100% and 95% confidence. Root cause traced by hand
(see the per-word probability dump this fix was validated against):
NOT any single "trigger word" - almost every word on a real page is
out-of-vocabulary for all four tiny hand-written classes, and for an
out-of-vocabulary word, Laplace/add-one smoothing gives
P(word|class) = 1/(total_words_in_class + vocab_size). Because the
four training classes had DIFFERENT total word counts, an
out-of-vocabulary word got a systematically higher smoothed
probability under the SMALLER classes, completely independent of the
page's actual content - so any sufficiently novel real page drifted
toward whichever class happened to have the smallest training corpus.
At the corpus sizes here (~50-90 words/class), that asymmetry isn't
negligible - it dominates. Two-part fix, both required:
  1. The training corpus below was rebalanced to comparable total word
     counts per class AND broadened with realistic vocabulary (crypto
     payment/privacy language, live-marketplace copy, generic infra
     wording) specifically covering the words that caused the two real
     false positives - so those words are no longer coincidentally
     absent from the "normal" class alone.
  2. A confidence/evidence gate (see "WHY 'uncertain' EXISTS") now
     catches whatever the corpus rebalance doesn't, so a genuinely
     novel page reports "uncertain" instead of a confident wrong label.
Both real cases are now regression tests in evaluate_content_signals.py
(REAL_WORLD_CASES) - reproduced there from the actual captured HTML,
not invented prose, and re-verified passing after this fix.

WHY "uncertain" EXISTS: with a training corpus this small, there will
always be real pages whose vocabulary the corpus simply doesn't cover
well, no matter how much the corpus above is broadened. Rather than
force one of the four labels on thin evidence, classify() now checks
three things before committing to a label:
  - MIN_SIGNAL_WORDS: at least this many distinct words in the page
    actually appeared somewhere in the training corpus (not just
    smoothing-floor probability for every class - real evidence).
  - CONFIDENCE_FLOOR: the winning class's probability clears a floor.
  - MARGIN_FLOOR: the winning class beats the runner-up by a real
    margin, not by a coin-flip-sized gap.
If any of those fail, the result is ("uncertain", <the winning class's
own probability, unchanged>, <its top_words, unchanged>) - the caller
still gets the classifier's honest lean and the words behind it (an
"opinion", not silence), it's just not asserted as a confident
category. monitor.py logs this straight to the CSV's content_type_guess
column, so "uncertain" is a visible, honest value there instead of a
guess dressed up as certainty.

HONEST LIMITATION: the training corpus below is still small and
hand-curated (now ~15-18 illustrative examples per category, not
thousands of real labeled pages), so this is a SECONDARY, corroborating
signal - it never overrides SEIZURE_BANNER_PATTERNS' direct keyword
match, which stays the higher-confidence primary signal for the exact
cases it covers. Treat a low/medium-confidence or "uncertain" classifier
result as "worth a human double-check," not as ground truth. Future
work: validate against a larger, real-world-labeled corpus (see
METHODOLOGY.md §8).
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Training corpus - intentionally small and fully readable in one pass, but
# rebalanced to comparable total word counts per class (see the module
# docstring's "REAL BUG FOUND AND FIXED HERE") so an out-of-vocabulary word
# doesn't get a systematically different smoothed probability purely because
# one class's corpus happens to be shorter than another's.
# seizure_banner examples are grounded in publicly documented real wording
# (Nemesis Market/BKA seizure banner, Operation Bayonet-style notices,
# generic DOJ/Europol/NCA seizure-notice phrasing seen in public reporting),
# plus terse title-only variants (no self-referential clause) matching a
# real miss found in production - see signal_utils.py's <title> pattern and
# evaluate_content_signals.py's REAL_WORLD_CASES. The other three categories
# are representative/illustrative examples of their category, not claimed
# real quotes from any specific site - normal_marketplace deliberately
# includes crypto-payment/privacy copy ("pay with bitcoin", "no logs no
# tracking", "end to end encryption") and generic infra wording ("our
# infrastructure handles..."), because those are exactly the words a real
# live marketplace or archive page uses and that a too-narrow corpus was
# previously missing.
# ---------------------------------------------------------------------------
TRAINING_CORPUS = {
    "seizure_banner": [
        "this platform has been seized by the federal criminal police office in frankfurt am main",
        "this domain has been seized by the fbi in accordance with a seizure warrant",
        "this hidden service has been seized as part of a coordinated law enforcement operation europol eurojust",
        "operation bayonet this website has been permanently shut down by federal agents and evidence secured",
        "in accordance with a forfeiture order issued by the department of justice this domain was seized",
        "national crime agency notice this marketplace has been taken down administrators arrested",
        "seized by homeland security investigations and the drug enforcement administration this site is no longer operational",
        "bundeskriminalamt notice dieser dienst wurde beschlagnahmt strafverfolgungsbehoerden",
        "seized by the federal bureau of investigation",
        "this domain name has been seized by the fbi",
        "notice to visitors this hidden service was shut down following a joint operation by international law enforcement agencies",
        "all servers seized funds frozen operators identified and charges filed in accordance with a judicial warrant",
        "warning unauthorized use of this domain following its seizure by federal authorities is a criminal offense",
        "this website and its contents have been seized pursuant to a court order issued by a district judge",
        "seizure of assets and domain by joint task force this platform is permanently offline effective immediately",
        "this server was seized as evidence during a law enforcement raid all user data has been secured",
    ],
    "error_maintenance": [
        "502 bad gateway nginx the server encountered a temporary error try again later",
        "site under maintenance please check back later we will be back soon sorry for the inconvenience",
        "503 service unavailable the server is currently unable to handle the request due to a temporary overload",
        "connection timed out the origin server took too long to respond please retry",
        "scheduled maintenance in progress upgrading our infrastructure please try again in a few hours",
        "internal server error something went wrong on our end our team has been notified",
        "cloudflare error 522 connection timed out between cloudflare and the origin web server",
        "this site is temporarily down for scheduled upgrades normal service will resume shortly",
        "we are experiencing technical difficulties please try again shortly our engineers are investigating the issue",
        "gateway timeout please retry in a few minutes if the problem persists contact support",
        "database connection failed the site will be back online once the issue is resolved",
        "this service is temporarily unavailable due to unexpected server load try again later",
        "planned downtime for infrastructure upgrades expected to last a few hours apologies for any inconvenience",
        "error 521 web server is down cloudflare could not reach the origin server",
        "we are performing routine maintenance and expect to be back online within the hour",
        "an unexpected error occurred while processing your request please refresh the page and try again",
    ],
    # Deliberately varies "new" with "updated"/"different"/"refreshed"/
    # "additional" across these examples - an earlier version of this list
    # used "new" in 13/16 examples, which (this small a corpus) made one of
    # the most common words in English the single strongest signal for this
    # category. Any page mentioning "new" ANYWHERE - a link directory
    # announcing a new mirror, a library site listing new books - got
    # pulled toward redesign_rebrand regardless of actual content.
    # Reproduced on two real captured pages (a Tor link directory and an
    # ebook library site, both confidently misclassified this way) before
    # rebalancing. "new" still appears twice below (genuine redesign
    # language legitimately uses it), just not in nearly every example.
    "redesign_rebrand": [
        "welcome to our new look same great vendors updated design and faster checkout",
        "we have relocated to a different domain please update your bookmarks and verify our updated pgp key",
        "announcing our rebrand updated name refreshed interface same trusted team and escrow system",
        "site relaunch overhauled escrow system improved search and category pages now live",
        "we are excited to unveil our redesigned marketplace with additional features and vendor tools",
        "important announcement updated mirror address please verify via our official pgp signed post before logging in",
        "after months of development we are proud to launch version two of our platform",
        "due to increased traffic we are migrating to different infrastructure please bookmark our updated address",
        "we just launched a brand new look for our marketplace faster pages and a cleaner design",
        "please note our vendor dashboard has been redesigned with additional analytics and order management tools",
        "after listening to community feedback we rebuilt our search and filtering system from scratch",
        "updated logo refreshed color scheme same reliable service please clear your cache to see the changes",
        "we upgraded our platform to a modern interface with improved mobile support",
        "important we changed our onion address again for security please verify the updated link via our pgp signed announcement",
        "celebrating one year online with a full site redesign additional features and a refreshed vendor directory",
        "our new homepage layout makes it easier to find top vendors and trending categories",
    ],
    "normal_marketplace": [
        "browse categories digital goods fraud drugs guides and tutorials search by keyword",
        "vendor listings escrow protected buyer feedback ratings verified sellers",
        "welcome back login register support tickets order history wallet balance",
        "featured listings top vendors this week trusted sellers free shipping",
        "search results filter by category price vendor rating shipping origin",
        "your cart order status shipping tracking dispute resolution center open a ticket",
        "vendor application form apply to sell on our marketplace review our seller guidelines",
        "frequently asked questions how to deposit how to withdraw how escrow works",
        "pay with bitcoin or monero for fast and private checkout no account required",
        "one hundred percent private no logs no tracking browse our catalog with confidence",
        "our platform uses end to end encryption and tor only access to protect your privacy",
        "fast delivery quick processing and encrypted ticket support available around the clock",
        "our infrastructure handles thousands of daily visitors across multiple mirror links for reliability",
        "secure checkout private messaging and pgp encrypted communication between buyers and sellers",
        "browse products by category add to cart and checkout securely with cryptocurrency",
        "new here read our faq to learn how deposits withdrawals and escrow protection work",
        # A link directory / wiki-style informational Tor site (not a literal
        # storefront) is a real, common site type this corpus otherwise has
        # no example of - and it happens to use ordinary words ("official",
        # "look", "pages", "important", "search") that redesign_rebrand's
        # announcement-style examples also lean on, which pulled two real
        # captured directory/library pages toward redesign_rebrand even
        # after rebalancing "new" above. Countering that the same way as
        # the earlier "hidden"/"secure"/"private" fix: give this vocabulary
        # real weight here too, not just in redesign_rebrand.
        "check out our directory of official links and category pages before you browse further",
        "this important listing page helps you look up verified links across many categories",
        "browse our search page to look up official mirror links across multiple categories",
        # Third-party seizure DISCUSSION, not a self-referential banner - added after
        # evaluate_content_signals.py measured 4/5 hard-negative false positives
        # (bag-of-words has no concept of "this site was seized" vs. "I heard that
        # OTHER site was seized" - both use the same words). This measurably helped
        # (4/5 -> 2/5 false positives on the same held-out hard-negative set) without
        # regressing the labeled test set, but did NOT eliminate the risk - see
        # METHODOLOGY.md §15 for the honest, still-real residual limitation.
        "heard that another market got seized by the fbi last month stay safe out there",
        "discussion thread about a rival platform getting shut down by law enforcement recently",
        "psa be careful of phishing clones impersonating markets that were seized",
        "some vendors lost their funds when that other forum got taken down by law enforcement last year always keep backups",
        "europol and other agencies take down a few unrelated markets every year always verify you are on the real link before logging in",
    ],
}

_WORD_RE = re.compile(r"[a-z]+")

# Excluded from BOTH classification and the explainability output
# (top_words) - with a corpus this small, common English function words
# score as "distinctive" purely by accident of which sentences happened to
# be written for which category (e.g. the seizure_banner examples above
# lean on "this X has been Y" phrasing more than the others do, so "this"
# alone was previously enough to pull real, unrelated informational pages
# toward seizure_banner - reproduced on a real captured page, see the
# module docstring). Originally excluded from top_words display only,
# with a comment claiming it never affected classification itself - that
# claim was wrong: _log_likelihood() summed over every token including
# stopwords. Fixed by excluding stopwords from scoring too, not just
# display, in both the training corpus and the page being classified -
# genre/topic classification should turn on content words, not sentence
# structure.
_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being to of by
for with as in on at from into and or but not no nor so if then than
it its it's we our you your they their he she his her i me my us them
please your you're we're has have had having
""".split())

# --- Evidence/confidence gate (see module docstring, "WHY 'uncertain' EXISTS") ---

MAX_TOKEN_REPEAT = 4     # cap any single repeated word's influence on the score
MIN_SIGNAL_WORDS = 3     # distinct non-stopword tokens actually seen in training data
CONFIDENCE_FLOOR = 0.55  # winning class must clear this probability
MARGIN_FLOOR = 0.12      # winning class must beat the runner-up by at least this much


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


def _content_tokenize(text):
    """Like _tokenize(), minus stopwords - see the _STOPWORDS comment
    above for why function words are excluded from scoring, not just
    from the top_words explanation."""
    return [w for w in _tokenize(text) if w not in _STOPWORDS]


class ContentTypeClassifier:
    """Multinomial Naive Bayes over word frequencies, trained once (at
    import time, on TRAINING_CORPUS above). `classify(text)` returns
    (best_label, probabilities_dict, top_words, signal_words) -
    top_words is what lets a caller show WHY, not just report a bare
    label; signal_words is the evidence-gate count used by
    classify_content_type() to decide whether to trust best_label at
    all or report "uncertain" instead."""

    def __init__(self, corpus):
        self.classes = list(corpus.keys())
        self.class_word_counts = {}
        self.class_totals = {}
        vocab = set()

        for label, examples in corpus.items():
            counts = Counter()
            for example in examples:
                counts.update(_content_tokenize(example))
            self.class_word_counts[label] = counts
            self.class_totals[label] = sum(counts.values())
            vocab.update(counts.keys())

        self.vocab_size = len(vocab)
        # Union of every word seen in ANY class's training data - used to
        # tell "genuine evidence" (a word the corpus actually knows,
        # regardless of which class) apart from "out-of-vocabulary word
        # that only looks like it's saying something because of smoothing."
        self._known_words = vocab
        n_docs = sum(len(examples) for examples in corpus.values())
        self.class_priors = {
            label: len(examples) / n_docs for label, examples in corpus.items()
        }

    def _log_likelihood(self, token_counts, label):
        # Laplace (add-one) smoothing (inside _word_prob) - an unseen word
        # shouldn't zero out the whole class.
        log_prob = math.log(self.class_priors[label])
        for word, count in token_counts.items():
            log_prob += count * math.log(self._word_prob(word, label))
        return log_prob

    def classify(self, text, max_words=400):
        all_tokens = _tokenize(text)[:max_words]
        tokens = [w for w in all_tokens if w not in _STOPWORDS]
        if not all_tokens:
            return None, {}, [], 0

        # Cap each distinct word's contribution (MAX_TOKEN_REPEAT) so one
        # repeated real word (a slogan, a nav label appearing in a footer
        # and a header) can't single-handedly dominate the product of
        # per-word likelihoods the way an uncapped raw count would.
        raw_counts = Counter(tokens)
        signal_words = sum(1 for w in raw_counts if w in self._known_words)

        # Score ONLY words the training corpus actually knows (appeared in
        # at least one class), not every out-of-vocabulary word via its
        # Laplace-smoothed floor. Found on a real long page (an ~800-word
        # informational archive site) that was confidently misclassified
        # despite having almost no real overlap with any class's
        # vocabulary: smoothing gives an out-of-vocabulary word
        # P=1/(total_class+V), which is NOT identical across classes when
        # each class's total training-word count differs even slightly -
        # so every single OOV word nudges the score toward whichever class
        # happens to have the smallest corpus, and on a long enough page
        # (hundreds of OOV words) that per-word nudge compounds into a
        # extreme, entirely content-free "confident" result. Restricting
        # scoring to known words makes an OOV word contribute exactly
        # nothing (correctly - it isn't evidence for any of the four
        # classes over the others), rather than a small biased nudge that
        # multiplies out of proportion on long pages.
        token_counts = Counter({
            w: min(c, MAX_TOKEN_REPEAT) for w, c in raw_counts.items() if w in self._known_words
        })

        log_probs = {label: self._log_likelihood(token_counts, label) for label in self.classes}

        # Convert log-probabilities to a normalized probability distribution
        # (subtracting the max first keeps math.exp() from overflowing/underflowing).
        max_log = max(log_probs.values())
        exp_probs = {label: math.exp(lp - max_log) for label, lp in log_probs.items()}
        total = sum(exp_probs.values())
        probabilities = {label: p / total for label, p in exp_probs.items()}

        best_label = max(probabilities, key=probabilities.get)

        # Words most DISTINCTIVE of the winning class, for explainability -
        # ranking by raw P(word|class) surfaces generic stopwords ("this",
        # "the", "of") that are common in every class, which would make the
        # "why" shown to an analyst actively misleading. Instead rank by how
        # much MORE likely each word is under the winning class than under
        # the average of the other classes - that surfaces words actually
        # characteristic of this category (e.g. "seized", "federal",
        # "warrant"), not just common English.
        other_labels = [l for l in self.classes if l != best_label]
        seen, token_scores = set(), []
        for word in tokens:
            if word in seen or word in _STOPWORDS:
                continue
            seen.add(word)
            p_best = self._word_prob(word, best_label)
            p_others = sum(self._word_prob(word, l) for l in other_labels) / len(other_labels)
            token_scores.append((word, p_best / p_others))
        top_words = [w for w, _ in sorted(token_scores, key=lambda x: -x[1])[:5]]

        return best_label, probabilities, top_words, signal_words

    def _word_prob(self, word, label):
        counts = self.class_word_counts[label]
        total = self.class_totals[label]
        return (counts.get(word, 0) + 1) / (total + self.vocab_size)


_classifier = ContentTypeClassifier(TRAINING_CORPUS)


def classify_content_type(text):
    """Public entry point. Returns (label, confidence, top_words):
    confidence is the winning class's probability (0.0-1.0), top_words
    are the words that most drove the decision. Always show top_words
    alongside the label - never the label alone - so a prediction stays
    auditable rather than an unexplained black-box guess.

    label is "uncertain" (not one of the four content categories) when
    the evidence is too thin to trust - see the module docstring's
    "WHY 'uncertain' EXISTS". confidence/top_words in that case are
    still the classifier's actual lean, not blanked out: a caller that
    wants the honest opinion behind the hedge can still read it, it's
    just not asserted as a confident category. Returns (None, 0.0, [])
    only for genuinely empty input (nothing to classify at all)."""
    if not text:
        return None, 0.0, []
    label, probs, top_words, signal_words = _classifier.classify(text)
    if label is None:
        return None, 0.0, []

    confidence = probs[label]
    sorted_probs = sorted(probs.values(), reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

    if signal_words < MIN_SIGNAL_WORDS or confidence < CONFIDENCE_FLOOR or margin < MARGIN_FLOOR:
        return "uncertain", confidence, top_words

    return label, confidence, top_words
