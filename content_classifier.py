"""
content_classifier.py
====================================================================
A small, from-scratch, interpretable text classifier (multinomial Naive
Bayes, trained on a curated corpus baked into this file) that buckets a
fetched page's content into one of four categories: seizure_banner,
error_maintenance, redesign_rebrand, normal_marketplace.

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

HONEST LIMITATION: the training corpus below is small and hand-curated
(a few illustrative examples per category, not thousands of real
labeled pages), so this is a SECONDARY, corroborating signal - it never
overrides SEIZURE_BANNER_PATTERNS' direct keyword match, which stays
the higher-confidence primary signal for the exact cases it covers.
Treat a low/medium-confidence classifier guess as "worth a human
double-check," not as ground truth. Future work: validate against a
larger, real-world-labeled corpus (see METHODOLOGY.md §8).
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Training corpus - intentionally small and fully readable in one pass.
# seizure_banner examples are grounded in publicly documented real wording
# (Nemesis Market/BKA seizure banner, Operation Bayonet-style notices,
# generic DOJ/Europol/NCA seizure-notice phrasing seen in public reporting).
# The other three categories are representative/illustrative examples of
# their category, not claimed real quotes from any specific site.
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
    ],
    "redesign_rebrand": [
        "welcome to our new look same great vendors updated design and faster checkout",
        "we have moved to a new domain please update your bookmarks and verify our new pgp key",
        "announcing our rebrand new name new interface same trusted team and escrow system",
        "site relaunch new escrow system improved search and category pages now live",
        "we are excited to unveil our redesigned marketplace with new features and vendor tools",
        "important announcement new mirror address please verify via our official pgp signed post before logging in",
        "after months of development we are proud to launch version two of our platform",
        "due to increased traffic we are migrating to new infrastructure please bookmark our new address",
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
    ],
}

_WORD_RE = re.compile(r"[a-z]+")

# Excluded from the explainability output (top_words) only, never from
# classification itself - with a corpus this small, common English function
# words can still score as "distinctive" by accident of which sentences
# happened to be written for which category (e.g. seizure examples above
# happen to lean on "this X has been Y" phrasing more than the others do).
# Showing "this, by, was, the" as the reason for a prediction would be
# actively misleading, not just unhelpful - this list keeps the shown
# "why" limited to words that are actually about content, not sentence
# structure.
_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being to of by
for with as in on at from into and or but not no nor so if then than
it its it's we our you your they their he she his her i me my us them
please your you're we're has have had having
""".split())


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


class ContentTypeClassifier:
    """Multinomial Naive Bayes over word frequencies, trained once (at
    import time, on TRAINING_CORPUS above). `classify(text)` returns
    (best_label, probabilities_dict, top_words) - top_words is what lets
    a caller show WHY, not just report a bare label."""

    def __init__(self, corpus):
        self.classes = list(corpus.keys())
        self.class_word_counts = {}
        self.class_totals = {}
        vocab = set()

        for label, examples in corpus.items():
            counts = Counter()
            for example in examples:
                counts.update(_tokenize(example))
            self.class_word_counts[label] = counts
            self.class_totals[label] = sum(counts.values())
            vocab.update(counts.keys())

        self.vocab_size = len(vocab)
        n_docs = sum(len(examples) for examples in corpus.values())
        self.class_priors = {
            label: len(examples) / n_docs for label, examples in corpus.items()
        }

    def _log_likelihood(self, tokens, label):
        # Laplace (add-one) smoothing (inside _word_prob) - an unseen word
        # shouldn't zero out the whole class.
        log_prob = math.log(self.class_priors[label])
        for word in tokens:
            log_prob += math.log(self._word_prob(word, label))
        return log_prob

    def classify(self, text, max_words=400):
        tokens = _tokenize(text)[:max_words]
        if not tokens:
            return None, {}, []

        log_probs = {label: self._log_likelihood(tokens, label) for label in self.classes}

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

        return best_label, probabilities, top_words

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
    auditable rather than an unexplained black-box guess. Returns
    (None, 0.0, []) for empty input."""
    if not text:
        return None, 0.0, []
    label, probs, top_words = _classifier.classify(text)
    if label is None:
        return None, 0.0, []
    return label, probs[label], top_words
