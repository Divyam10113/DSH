"""Rule/keyword baseline labeler for the 12 findings.

Each finding is described by one or more `Rule`s. A rule is a set of regex groups that must all
occur in the same sentence of the (accent-folded) report; group 0 is the *anchor*. Companion groups
must lie within `max_dist` tokens of the anchor. Each hit is then classified as

    affirmed  -> 1.0
    uncertain -> 0.5   ("possible", "Verdacht auf", "no se descarta", ...)
    negated   -> 0.0   ("no", "kein", "sin", "intact", "unauffällig", ...)

in the spirit of NegEx. A report's score for a label is the max over its hits (0 if never mentioned).

Covers English, German, Spanish, French, Italian, Portuguese and Dutch. Patterns are written
against `fold_text` output: lowercase, no accents, "ß" -> "ss", "œ" -> "oe".
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ID_COL, LABELS, REPORT_COL
from .text import fold_text, split_sentences


def _any(*patterns: str) -> str:
    return "(?:" + "|".join(patterns) + ")"


# ---------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------

LIGAMENT_INJURY = _any(
    r"\btear", r"\btorn", r"rupt", r"\bsprain", r"\binjur", r"\bdisrupt", r"discontinu",
    r"\blesion", r"\bpartial", r"insufficien", r"deficien", r"\bavuls", r"\bstrain",
    r"riss", r"lasion", r"verletz", r"zerrung", r"distorsion",          # de
    r"\brotura", r"desgarr", r"esguince",                                # es / pt
    r"dechir", r"entorse",                                               # fr / pt
    r"rottur", r"lesione", r"lacera", r"distrazion",                     # it
    r"lesao", r"laceracao",                                              # pt
    r"scheur", r"letsel", r"laesie", r"ruptuur",                         # nl
)

MENISCUS_TEAR = _any(
    r"\btear", r"\btorn", r"rupt", r"fissur", r"\blesion", r"\bflap", r"bucket[\s-]?handle",
    r"riss", r"lasion", r"korbhenkel", r"lappen",                        # de
    r"\brotura", r"desgarr",                                             # es / pt
    r"dechir", r"anse\s+de\s+seau",                                      # fr
    r"rottur", r"lesione", r"lacera", r"manico\s+di\s+secchio",          # it
    r"lesao", r"laceracao", r"alca\s+de\s+balde",                        # pt
    r"scheur", r"laesie", r"ruptuur",                                    # nl
)

ACL = _any(
    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"anterior\s+cruciate", r"cruciate\s+ligament\s+anterior",
    r"vordere\w*\s+kreuzband", r"cruzado\s+anterior", r"croise\s+anterieur",
    r"crociato\s+anteriore", r"voorste\s+kruisband", r"cruciatum\s+anterius",
)

MCL = _any(
    r"\bmcl\b", r"\blcm\b", r"\blli\b", r"medial\s+collateral", r"tibial\s+collateral",
    r"innenband", r"medial\w*\s+(?:kollateral|seiten)band", r"(?:kollateral|seiten)band\w*\s+medial",
    r"colateral\s+medial", r"collateral\s+medial", r"collaterale\s+mediale",
    r"lateral\s+interno", r"lateral\s+interne", r"mediale\s+collaterale\s+band", r"binnenband",
)

MENISCUS = _any(r"menisc", r"menisk", r"menisq")

# Side words for the meniscus. A side word directly qualifying another structure ("lateral meniscus
# and medial femoral condyle") does not count. The trailing \b stops \w* backtracking past it.
_QUALIFIES_OTHER = (
    r"(?!\s*(?:femor|tibia|condyl|condil|plateau|plato|compart|kompart|facet|patell|collat|kollat"
    r"|retinac|joint|gelenk|gastrocn|head|kopf))"
)
MEDIAL = _any(*(p + _QUALIFIES_OTHER for p in (
    r"\bmedial\w*\b", r"\binterno\b", r"\binterne\b", r"\binnen(?!band)\w*\b")))
LATERAL = _any(*(p + _QUALIFIES_OTHER for p in (
    r"\blateral\w*\b", r"\bexterno\b", r"\bexterne\b", r"\baussen(?!band)\w*\b")))
# For OA, a side word must describe a tibiofemoral compartment, not "medial meniscus/patellar facet".
_NOT_COMPARTMENT = r"(?!\s*(?:patell|facet|retinac|collat|kollat|seitenband|plica|menisc|menisk|menisq))"
MEDIAL_COMPARTMENT = _any(*(p + _NOT_COMPARTMENT for p in (
    r"\bmedial\w*\b", r"\binterno\b", r"\binterne\b", r"\binnen(?!band|menisk)\w*\b")))
LATERAL_COMPARTMENT = _any(*(p + _NOT_COMPARTMENT for p in (
    r"\blateral\w*\b", r"\bexterno\b", r"\bexterne\b", r"\baussen(?!band|menisk)\w*\b")))

OA = _any(
    r"osteoarthr", r"arthros(?![ck])", r"artros(?![ck])", r"gonarthr", r"degenerative\s+change",
    r"chondr\w*\s+(?:loss|thinning|defect|wear)", r"cartilag\w*\s+(?:loss|thinning|defect|wear)",
    r"chondropath", r"chondromalac", r"condromalac", r"condropat", r"osteophyt", r"osteofit",
    r"osteofyt", r"joint\s+space\s+narrow", r"knorpel\w*\s*(?:schad|verschmal|verlust|glatze|defekt)",
    r"perdida\s+de\s+cartilago", r"perte\s+cartilagin", r"usura\s+cartilag", r"condral",
)
PATELLOFEMORAL = _any(
    r"patell", r"femoro[\s-]?patell", r"patel\w*[\s-]?femor", r"femoropatelar", r"retropatell",
    r"trochle", r"\brotul", r"kniescheibe",
)
TRICOMPARTMENT = _any(r"tri[\s-]?compart", r"trikompart", r"pangonarthr", r"all\s+(?:three\s+)?compartments")

EFFUSION = _any(
    r"effusion", r"erguss", r"derrame", r"epanchement", r"versamento", r"hydrops", r"hydrarthr",
    r"idrarto", r"effusie",
)
SYNOVITIS = _any(
    r"synovit", r"synovialit", r"sinovit",
    r"synovi\w*\s+(?:thicken|hypertroph|prolifer)", r"sinovi\w*\s+(?:ispessi|engrosa|hipertrof|espessa)",
)
BAKER = _any(
    r"\bbaker", r"poplit\w*\s*(?:cyst|zyst)", r"(?:cyst|zyst|quiste|kyste|cisti|cisto|quisto)\w*\s+(?:\w+\s+)?poplit",
    r"gastrocnemi\w*[\s-]+semimembran", r"semimembran\w*[\s-]+gastrocnemi",
)
CONTUSION = _any(
    r"contus", r"kontus", r"bone\s+bruis", r"(?:bone\s+)?marrow\s+o?edema", r"knochenmark\w*\s*odem",
    r"edema\s+(?:de\s+la\s+)?(?:medula\s+)?osea", r"edema\s+oseo", r"o?edeme\s+(?:osseux|medullaire)",
    r"edema\s+(?:osseo|midollare)", r"edema\s+(?:da\s+medula\s+)?ossea", r"beenmerg\s*o?edeem",
    r"botkneuz",
)
FRACTURE = _any(r"fractu", r"fraktur", r"frattur", r"fratur", r"\bfx\b", r"\bbreuk")

# Negation / uncertainty cues (NegEx-style).
PRE_NEGATION = re.compile(r"\b" + _any(
    r"no", r"not", r"without", r"negative\s+for", r"absence\s+of", r"free\s+of", r"neither", r"nor",
    r"kein\w*", r"nicht", r"ohne", r"weder",
    r"sin", r"ausencia\s+de", r"ni",
    r"pas\s+d", r"pas\s+de", r"sans", r"absence\s+d", r"aucun\w*",
    r"non", r"senza", r"assenza\s+di", r"ne",
    r"nao", r"sem", r"nem",
    r"geen", r"niet", r"zonder",
) + r"\b")
POST_NEGATION = re.compile(r"\b" + _any(
    r"intact", r"normal", r"none", r"nil", r"negative", r"unremarkable", r"preserved", r"absent",
    r"not\s+(?:seen|identified|present)",
    r"intakt", r"unauffallig", r"regelrecht", r"normal\w*", r"erhalten", r"nicht", r"keiner?",
    r"integr[oa]", r"conservad[oa]", r"ausente", r"respecte\w*", r"conservat[oa]", r"assente",
    r"preservad[oa]", r"normaal", r"geen",
) + r"\b")
UNCERTAIN = re.compile(r"\b" + _any(
    r"possibl\w*", r"probabl\w*", r"questionabl\w*", r"suspect\w*", r"suspicious", r"equivocal",
    r"cannot\s+be\s+(?:excluded|ruled\s+out)", r"not\s+excluded", r"may\s+represent", r"rule\s+out",
    r"verdacht", r"fraglich\w*", r"moglich\w*", r"nicht\s+auszuschliessen",
    r"posible", r"sospech\w*", r"no\s+se\s+descarta", r"suspicion", r"douteu\w*",
    r"possibile", r"sospett\w*", r"dubbi\w*", r"possivel", r"suspeit\w*", r"provavel",
    r"mogelijk", r"verdenking",
) + r"\b")
# Words that end a negation scope ("no fracture, but effusion"; "no fracture and small effusion").
TERMINATORS = re.compile(r"\b" + _any(
    r"but", r"however", r"although", r"and", r"with", r"aber", r"jedoch", r"und", r"mit",
    r"pero", r"con", r"y", r"mais", r"cependant", r"et", r"avec", r"ma", r"tuttavia", r"e",
    r"mas", r"porem", r"com", r"maar", r"echter", r"met",
) + r"\b")

PRE_WINDOW = 6         # tokens before a matched term searched for a negation cue ("no ACL tear")
POST_WINDOW = 3        # tokens after the hit searched for a post-negation cue ("effusion: none")
UNCERTAIN_WINDOW = 6   # tokens either side of the hit searched for a hedge ("possible tear")


@dataclass(frozen=True)
class Rule:
    label: str
    groups: tuple[str, ...]
    max_dist: tuple[int, ...] = ()   # per companion group; defaults to 12 tokens

    def compiled(self) -> list[re.Pattern]:
        return [re.compile(g) for g in self.groups]


def _oa_rules(label: str, side: str) -> list[Rule]:
    return [Rule(label, (OA, side), (8,)), Rule(label, (OA, TRICOMPARTMENT), (8,))]


RULES: list[Rule] = [
    Rule("ACL", (ACL, LIGAMENT_INJURY), (8,)),
    Rule("MCL", (MCL, LIGAMENT_INJURY), (8,)),
    Rule("Medial Meniscus", (MENISCUS, MEDIAL, MENISCUS_TEAR), (4, 12)),
    Rule("Lateral Meniscus", (MENISCUS, LATERAL, MENISCUS_TEAR), (4, 12)),
    *_oa_rules("Medial OA", MEDIAL_COMPARTMENT),
    *_oa_rules("Lateral OA", LATERAL_COMPARTMENT),
    *_oa_rules("PF OA", PATELLOFEMORAL),
    Rule("Effusion", (EFFUSION,)),
    Rule("Synovitis", (SYNOVITIS,)),
    Rule("Baker's", (BAKER,)),
    Rule("Contusion", (CONTUSION,)),
    Rule("Fracture", (FRACTURE,)),
]

AFFIRMED, UNCERTAIN_HIT, NEGATED = "affirmed", "uncertain", "negated"
_HIT_SCORE = {AFFIRMED: 1.0, UNCERTAIN_HIT: 0.5, NEGATED: 0.0}


class _Sentence:
    """A sentence with token offsets, so windows can be measured in tokens."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = [m.span() for m in re.finditer(r"\w+", text)]
        self._starts = [s for s, _ in self.tokens]

    def token_index(self, char_pos: int) -> int:
        return max(bisect.bisect_right(self._starts, char_pos) - 1, 0)

    def before(self, char_pos: int, n_tokens: int) -> str:
        """Up to `n_tokens` whole tokens ending before `char_pos`, cut after the last terminator."""
        first = bisect.bisect_left(self._starts, char_pos)  # tokens [0, first) start before char_pos
        if first == 0:
            return ""
        text = self.text[self.tokens[max(first - n_tokens, 0)][0]:char_pos]
        last = None
        for last in TERMINATORS.finditer(text):
            pass
        return text[last.end():] if last else text

    def after(self, char_pos: int, n_tokens: int) -> str:
        """Up to `n_tokens` tokens starting at `char_pos`, cut at the first comma or terminator."""
        first = bisect.bisect_left(self._starts, char_pos)
        if first >= len(self.tokens):
            return ""
        text = self.text[char_pos:self.tokens[min(first + n_tokens, len(self.tokens)) - 1][1]]
        text = re.split(r"[,(]", text, maxsplit=1)[0]
        m = TERMINATORS.search(text)
        return text[:m.start()] if m else text


def _classify_hit(sent: _Sentence, spans: list[tuple[int, int]]) -> str:
    """Affirmed / uncertain / negated for a hit made of the matched term spans."""
    left = min(s for s, _ in spans)
    right = max(e for _, e in spans)

    around = sent.before(left, UNCERTAIN_WINDOW) + " " + sent.after(right, UNCERTAIN_WINDOW)
    if UNCERTAIN.search(around) or UNCERTAIN.search(sent.text[left:right]):
        return UNCERTAIN_HIT

    # A pre-negation in front of any matched term negates the hit: "no ACL tear",
    # "ACL with no evidence of tear", "kein Innenmeniskusriss".
    for start, _ in spans:
        if PRE_NEGATION.search(sent.before(start, PRE_WINDOW)):
            return NEGATED
    if POST_NEGATION.search(sent.after(right, POST_WINDOW)):
        return NEGATED
    return AFFIRMED


def _nearest(pat: re.Pattern, sent: _Sentence, anchor_tok: int) -> tuple[int, tuple[int, int]] | None:
    candidates = [(abs(sent.token_index(m.start()) - anchor_tok), m.span()) for m in pat.finditer(sent.text)]
    return min(candidates) if candidates else None


def _rule_hits(rule: Rule, patterns: list[re.Pattern], sent: _Sentence) -> list[str]:
    hits = []
    for anchor in patterns[0].finditer(sent.text):
        anchor_tok = sent.token_index(anchor.start())
        spans = [anchor.span()]
        for i, pat in enumerate(patterns[1:]):
            max_dist = rule.max_dist[i] if i < len(rule.max_dist) else 12
            best = _nearest(pat, sent, anchor_tok)
            if best is None or best[0] > max_dist:
                break
            spans.append(best[1])
        else:
            hits.append(_classify_hit(sent, spans))
    return hits


_COMPILED = [(rule, rule.compiled()) for rule in RULES]


def extract_report(report: str) -> dict[str, dict[str, int]]:
    """Per-label hit counts for one report: {label: {"affirmed": n, "uncertain": n, "negated": n}}."""
    counts = {label: {AFFIRMED: 0, UNCERTAIN_HIT: 0, NEGATED: 0} for label in LABELS}
    for sentence in split_sentences(fold_text(report)):
        sent = _Sentence(sentence)
        for rule, patterns in _COMPILED:
            for hit in _rule_hits(rule, patterns, sent):
                counts[rule.label][hit] += 1
    return counts


def score_counts(counts: dict[str, int]) -> float:
    """Collapse hit counts to a label score: any affirmed -> 1, else any uncertain -> 0.5, else 0."""
    if counts[AFFIRMED]:
        return _HIT_SCORE[AFFIRMED]
    if counts[UNCERTAIN_HIT]:
        return _HIT_SCORE[UNCERTAIN_HIT]
    return 0.0


def rule_features(reports: pd.Series) -> pd.DataFrame:
    """Hit counts per label and hit type ("ACL|affirmed", ...), one row per report (index preserved).

    Used both to derive rule scores and as extra features for the text classifier.
    """
    rows = []
    for r in reports:
        counts = extract_report(r)
        rows.append({f"{label}|{kind}": n for label, c in counts.items() for kind, n in c.items()})
    columns = [f"{label}|{kind}" for label in LABELS for kind in (AFFIRMED, UNCERTAIN_HIT, NEGATED)]
    return pd.DataFrame(rows, index=reports.index, columns=columns).astype(np.float32)


def scores_from_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Rule score in {0, 0.5, 1} per label from `rule_features` output."""
    out = pd.DataFrame(index=feats.index, columns=LABELS, dtype=float)
    for label in LABELS:
        out[label] = np.where(feats[f"{label}|{AFFIRMED}"] > 0, 1.0,
                              np.where(feats[f"{label}|{UNCERTAIN_HIT}"] > 0, 0.5, 0.0))
    return out


def rule_scores(reports: pd.Series) -> pd.DataFrame:
    """Rule score in {0, 0.5, 1} for every label, one row per report (index preserved)."""
    return scores_from_features(rule_features(reports))


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: rule scores for a train.csv-shaped frame, keyed by StudyInstanceUID."""
    scores = rule_scores(df[REPORT_COL].fillna(""))
    scores.insert(0, ID_COL, df[ID_COL].values)
    return scores
