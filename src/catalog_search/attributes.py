"""Conservative, deterministic English catalog attributes.

``extract_attributes`` returns JSON-serializable records with original-text spans,
canonical values/units, a comparison operator, semantic role and uncertainty flags.
``compare_attributes`` returns fixed-order numerical features and per-query evidence.
An agreement concerns an attribute, not overall product relevance or physical safety.
Missing, differently scoped and ambiguous specifications are ``unknown``.

Only explicit units, known model-family grammars and a small explicit brand lexicon
are parsed. Arbitrary numerals are never requirements. GB/TB use decimal SI prefixes;
GiB/TiB use binary prefixes; inches convert to exactly 25.4 mm. Close but unequal
storage capacities remain unknown because catalogs often mix these conventions.
Different nominal wattages alone cannot establish charger incompatibility: USB-PD
negotiates supported power (https://www.usb.org/usb-charger-pd). Explicit bounds are
compared as written. SI reference: https://www.nist.gov/pml/owm/metric-si-prefixes.

This module has no catalog-trained vocabulary and does not use relevance labels.
It deliberately abstains on ambiguous bundles, alternatives, dimensions without
axis labels, unsupported model grammars and unverified manufacturer identity.
"""

from __future__ import annotations

import math
import re
from typing import Any

KINDS = ("storage", "power", "length", "model", "brand")
FEATURE_NAMES = tuple(
    [
        "attr_query_count", "attr_product_count", "attr_match_count",
        "attr_conflict_count", "attr_unknown_count", "attr_match_fraction",
        "attr_conflict_fraction", "attr_unknown_fraction", "attr_has_conflict",
        "attr_ambiguous_query_count",
    ]
    + [f"attr_{kind}_{suffix}" for kind in KINDS
       for suffix in ("query_count", "match_count", "conflict_count", "unknown_count")]
)

_NUMBER = r"(?:\d+(?:\s*\+\s*|\s+(?:and\s+)?)\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d*[¼½¾⅛⅜⅝⅞]|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
_UNIT = r'(?:tib|gib|mib|tb|gb|mb|kilowatts?|kw|watts?|w|millimeters?|millimetres?|mm|centimeters?|centimetres?|cm|inches|inch|in\.?|meters?|metres?|m|[\"″])'
_RANGE = re.compile(
    rf'(?<![\w.])(?P<lo>{_NUMBER})\s*(?P<u1>{_UNIT})?\s*'
    rf'(?P<separator>[-–—]|\bto\b|\band\b)\s*(?P<hi>{_NUMBER})\s*(?P<unit>{_UNIT})(?![\w])',
    re.I,
)
_SCALAR = re.compile(rf'(?<![\w.])(?P<lo>{_NUMBER})\s*[-]?\s*(?P<unit>{_UNIT})(?![\w])', re.I)
_BRANDS = {
    "samsung": "samsung", "sony": "sony", "lenovo": "lenovo", "dell": "dell",
    "asus": "asus", "acer": "acer", "anker": "anker", "belkin": "belkin",
    "ugreen": "ugreen", "sandisk": "sandisk", "kingston": "kingston",
    "seagate": "seagate", "western digital": "western digital", "logitech": "logitech",
    "canon": "canon", "nikon": "nikon", "epson": "epson", "nintendo": "nintendo",
    "motorola": "motorola", "oneplus": "oneplus", "dewalt": "dewalt",
    "makita": "makita", "bosch": "bosch", "dyson": "dyson", "kitchenaid": "kitchenaid",
    "apple": "apple", "google": "google", "amazon": "amazon", "microsoft": "microsoft",
}
_BRAND_RE = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in sorted(_BRANDS, key=len, reverse=True)) + r")\b", re.I)
_AMBIGUOUS_BRAND_CONTEXT = {
    "apple": r"\b(?:iphone|ipad|macbook|imac|airpods|ipod|watch|charger|pencil|keyboard|tv)\b",
    "google": r"\b(?:pixel|chromecast|nest|chromebook|phone)\b",
    "amazon": r"\b(?:kindle|echo|fire|tablet|basics|essentials)\b",
    "microsoft": r"\b(?:surface|xbox|windows|office|keyboard|mouse)\b",
}
_ACCESSORY = re.compile(r"\b(?:cases?|covers?|chargers?|cables?|adapters?|protectors?|replacement|sleeves?|stands?|mounts?)\b", re.I)
_COMPAT = re.compile(r"(?:\b(?:for|fits?|supports?|compatible(?:\s+with)?|replacement\s+for)\s+)(?:[\w-]+\s+){0,3}$", re.I)
_FAMILIES = (
    ("iphone", re.compile(r"\biphone\s*-?\s*(?:\d{1,2}|se|xr|xs|x)(?:\s*(?:pro|max|plus|mini)){0,2}\b", re.I)),
    ("galaxy", re.compile(r"\bgalaxy\s+(?:(?:s|a|m|z)\s*\d{1,3}|note\s*\d{1,2}|(?:fold|flip)\s*\d)(?:\s*(?:ultra|plus|fe|5g)){0,2}\b", re.I)),
    ("pixel", re.compile(r"\bpixel\s+\d{1,2}a?(?:\s*(?:pro|xl|fold)){0,2}\b", re.I)),
    ("surface", re.compile(r"\bsurface\s+(?:pro\s*\d{1,2}|laptop\s*\d{1,2}|go\s*\d)(?:\s*plus)?\b", re.I)),
    ("playstation", re.compile(r"\b(?:playstation\s*|ps)\d(?:\s*(?:pro|slim))?\b", re.I)),
)
_SKU = re.compile(r"\b[A-Za-z]{1,8}-?\d[A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*\b")
_NON_MODELS = re.compile(r"^(?:\d|\d+[a-z]*|[tgmk]i?b|\d+(?:pack|pcs|piece|in|mm|cm|w|k|v|hz)|(?:usb|wifi|bluetooth|merv|ddr|ip|hdmi|pcie|cat)\d.*|[24]in1|[248]k)$", re.I)


def _num(value: str) -> float:
    fractions = {"¼": .25, "½": .5, "¾": .75, "⅛": .125, "⅜": .375, "⅝": .625, "⅞": .875}
    if value[-1] in fractions:
        return float(value[:-1] or 0) + fractions[value[-1]]
    if "/" in value:
        numbers = [int(n) for n in re.findall(r"\d+", value)]
        if len(numbers) == 2:
            return numbers[0] / numbers[1]
        return numbers[0] + numbers[1] / numbers[2]
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", value) and not value.startswith("0,"):
        return float(value.replace(",", ""))
    return float(value.replace(",", "."))


def _unit(unit: str) -> tuple[str, str, float]:
    u = unit.lower().rstrip(".")
    storage = {"mb": 1e6, "gb": 1e9, "tb": 1e12,
               "mib": float(2**20), "gib": float(2**30), "tib": float(2**40)}
    if u in storage:
        return "storage", "byte", storage[u]
    if u in {"w", "watt", "watts", "kw", "kilowatt", "kilowatts"}:
        return "power", "watt", 1000.0 if u.startswith("k") else 1.0
    if u in {'"', "″", "in", "inch", "inches"}:
        return "length", "mm", 25.4
    if u.startswith("cm") or u.startswith("centim"):
        return "length", "mm", 10.0
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return "length", "mm", 1000.0
    return "length", "mm", 1.0


def _clause_prefix(text: str, start: int) -> str:
    return re.split(r"[.;!?\n]|\b(?:but|however)\b", text[max(0, start - 65):start], flags=re.I)[-1]


def _operator(text: str, start: int, end: int) -> tuple[str, bool]:
    pre = _clause_prefix(text, start).lower()
    post = text[end:end + 22].lower()
    if re.search(r"(?:\b(?:not|without|excluding|exclude|except|no)\s+(?:(?:compatible|suitable)\s+(?:with|for)\s+|for\s+)?)$", pre):
        return "neq", True
    if re.search(r"(?:at\s+least|minimum(?:\s+of)?|no\s+less\s+than|>=|≥)\s*$", pre) or re.match(r"\s*(?:or\s+(?:more|higher|larger)|minimum)\b", post):
        return "ge", True
    if re.search(r"(?:up\s+to|at\s+most|maximum(?:\s+of)?|no\s+more\s+than|<=|≤)\s*$", pre) or re.match(r"\s*(?:or\s+(?:less|lower|smaller)|maximum)\b", post):
        return "le", True
    if re.search(r"(?:more\s+than|over|>|greater\s+than)\s*$", pre):
        return "gt", True
    if re.search(r"(?:less\s+than|under|<|below)\s*$", pre):
        return "lt", True
    return "eq", bool(re.search(r"\bexactly\s*$", pre))


def _role(text: str, start: int, end: int, kind: str) -> str:
    before = text[max(0, start - 28):start].lower()
    after = text[end:end + 28].lower()
    nearby = before + " " + after
    if kind == "storage":
        if re.match(r"\s*(?:/\s*s(?:ec(?:ond)?s?)?\b|per\s+second\b)", after):
            return "transfer_rate"
        if re.match(r"\s*(?:of\s+)?ram\b", after) or re.search(r"\bram\s*[:=]?\s*$", before):
            return "ram"
        if re.match(r"\s*(?:total|combined)\b", after) or re.search(r"\b(?:total|combined)\s*[:=]?\s*$", before):
            return "bundle_total"
        return "capacity"
    if kind == "power":
        if re.match(r"\s*(?:equivalent|replacement)\b", after) or re.search(r"\b(?:equivalent\s+to|replaces)\s*$", before):
            return "equivalent"
        for name in ("input", "output"):
            if re.search(rf"\b{name}\s*(?:power\s*)?[:=]?\s*$", before) or re.match(rf"\s*{name}\b", after):
                return name
        return "rated"
    if kind == "length":
        for role, pattern in (
            ("width", r"width|wide"), ("height", r"height|tall"),
            ("diameter", r"diameter|dia"), ("length", r"length|long"),
        ):
            if re.search(rf"\b(?:{pattern})\s*[:=]?\s*$", before) or re.match(rf"\s*(?:{pattern})\b", after):
                return role
        if re.search(r"\b(?:screen|display|monitor|laptop|tablet|tv|macbook)\b", nearby):
            return "screen"
        return "unspecified"
    return "unspecified"


def _ambiguity(text: str, start: int, end: int, kind: str) -> tuple[bool, str | None]:
    pre, post = text[max(0, start - 22):start], text[end:end + 24]
    if re.search(rf"{_NUMBER}\s*(?:{_UNIT})?\s*(?:[x×*]|/)\s*$", pre, re.I) or re.match(r"\s*(?:[x×*]|/)\s*\d", post):
        return True, "bundle, dimensions or alternatives have unresolved scope"
    if re.search(r"\bor\s*$", pre, re.I) or re.match(r"\s*or\b", post, re.I):
        return True, "alternative requirements are not conjunctive constraints"
    if pre.rstrip().endswith("/"):
        return True, "incomplete fraction or unresolved slash expression"
    if kind == "model" and re.match(r"\s*[/,]\s*\d", post):
        return True, "elided model list is not fully parsed"
    return False, None


def _record(text: str, start: int, end: int, **values: Any) -> dict[str, Any]:
    return {"text": text[start:end], "span": [start, end], "confidence": "high", **values}


def extract_attributes(text: str) -> list[dict[str, Any]]:
    """Extract conservative typed attributes; spans always index the supplied text.

    Numeric ``value/lower/upper`` use the canonical unit. ``operator`` is eq, neq,
    ge, gt, le, lt or range; absent interval endpoints are ``None``. ``coverage``
    distinguishes an explicit supported range ("fits 13–15 inch") from an uncertain
    measured range. ``ambiguous`` records are evidence, but never hard conflicts.
    Brand recognition is intentionally incomplete; there is no general NER fallback.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    found: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for regex in (_RANGE, _SCALAR):
        for match in regex.finditer(text):
            start, end = match.span()
            if any(start < b and end > a for a, b in occupied):
                continue
            raw_unit = match.group("unit")
            kind, canonical_unit, multiplier = _unit(raw_unit)
            # Bare "in" also occurs in prose ("made in 2020", "2 in 1").
            if raw_unit.lower().rstrip(".") == "in" and re.match(r"\s+\d", text[end:]):
                continue
            try:
                lo = _num(match.group("lo")) * multiplier
                range_hi = _num(match.group("hi")) * multiplier if "hi" in match.groupdict() else lo
            except ZeroDivisionError:
                continue
            hi = lo
            op, explicit = _operator(text, start, end)
            is_range = "hi" in match.groupdict()
            ambiguous, reason = _ambiguity(text, start, end, kind)
            if kind == "length" and lo <= 0:
                ambiguous, reason = True, "non-positive dimension is not a usable size constraint"
            width_notation = raw_unit.lower() == "w" and bool(re.match(r"\s*(?:[x×*]\s*)?\d+(?:\.\d+)?\s*[lhd]\b", text[end:], re.I))
            if width_notation:
                ambiguous, reason = True, "W next to L/H/D may denote width rather than electrical power"
            if raw_unit in {"Mb", "Gb", "Tb", "Mib", "Gib", "Tib"}:
                ambiguous, reason = True, "lowercase b may denote bits rather than bytes"
            if kind == "length" and raw_unit == "M":
                ambiguous, reason = True, "uppercase M may be a size code rather than metres"
            if is_range:
                if op == "neq":
                    ambiguous, reason = True, "negated range is not a positive requirement interval"
                mixed_fraction = match.group("separator").lower() == "and" and "/" in match.group("hi") and not match.group("u1")
                if match.group("separator").lower() == "and" and not mixed_fraction and not re.search(r"\bbetween\s*$", _clause_prefix(text, start), re.I):
                    ambiguous, reason = True, "a conjunction of values is not necessarily a range"
                first_unit = match.group("u1")
                if first_unit:
                    first_kind, _, first_multiplier = _unit(first_unit)
                    lo = _num(match.group("lo")) * first_multiplier
                    if first_kind != kind:
                        ambiguous, reason = True, "range combines different dimensions"
                hi = range_hi
                if lo > hi and not mixed_fraction:
                    ambiguous, reason = True, "descending range"
                op, explicit = "range", True
                if mixed_fraction:
                    lo = hi = lo + hi
                    op, explicit = _operator(text, start, end)
            lower = None if op in {"le", "lt"} else lo
            upper = None if op in {"ge", "gt"} else hi
            pre = _clause_prefix(text, start)
            coverage = bool(_COMPAT.search(pre))
            role = _role(text, start, end, kind)
            if role == "transfer_rate":
                canonical_unit = "byte/s"
            found.append(_record(
                text, start, end, kind=kind, value=lo, unit=canonical_unit,
                raw_unit=raw_unit, lower=lower, upper=upper, operator=op,
                explicit=explicit, role=role,
                coverage=coverage, ambiguous=ambiguous, uncertainty=reason,
            ))
            occupied.append((start, end))
    for family, regex in _FAMILIES:
        for match in regex.finditer(text):
            start, end = match.span()
            value = re.sub(r"[\s-]+", "", match.group().lower())
            if family == "playstation":
                value = re.sub(r"^ps", "playstation", value)
            op, explicit = _operator(text, start, end)
            ambiguous, reason = _ambiguity(text, start, end, "model")
            found.append(_record(
                text, start, end, kind="model", value=value, unit=None,
                operator=op, explicit=explicit, role=family,
                coverage=bool(_COMPAT.search(_clause_prefix(text, start))),
                ambiguous=ambiguous, uncertainty=reason,
            ))
            occupied.append((start, end))
    for match in _SKU.finditer(text):
        start, end = match.span()
        raw = match.group()
        if len(raw) < 4 or _NON_MODELS.fullmatch(raw) or any(start < b and end > a for a, b in occupied):
            continue
        # A mixed alphanumeric token is an identifier signal, not every number.
        canonical = re.sub(r"[-.]", "", raw.lower())
        family = re.match(r"[A-Za-z]+", canonical).group().lower()
        explicit_model = bool(re.search(r"\b(?:model|part|sku)(?:\s*(?:no|number))?\s*[:#=]?\s*$", _clause_prefix(text, start), re.I))
        # Long ordinary words joined to a number are often tokenization artifacts.
        # Short prefixes, internal separators and explicit model fields retain IDs.
        if len(family) > 3 and "-" not in raw and not raw.isupper() and not explicit_model:
            continue
        op, explicit = _operator(text, start, end)
        ambiguous, reason = _ambiguity(text, start, end, "model")
        found.append(_record(text, start, end, kind="model", value=canonical,
                             unit=None, operator=op, explicit=explicit,
                             role="sku:" + family, coverage=False,
                             ambiguous=ambiguous, uncertainty=reason))
    for match in _BRAND_RE.finditer(text):
        start, end = match.span()
        brand = _BRANDS[match.group().lower()]
        if brand in _AMBIGUOUS_BRAND_CONTEXT and not re.search(_AMBIGUOUS_BRAND_CONTEXT[brand], text, re.I) and not re.search(r"\bbrand\s*[:=]\s*$", _clause_prefix(text, start), re.I):
            continue
        op, explicit = _operator(text, start, end)
        compatibility = bool(_COMPAT.search(_clause_prefix(text, start)))
        role = "compatibility" if compatibility else "manufacturer"
        if brand == "microsoft" and re.match(r"\s+(?:office|windows)\b", text[end:], re.I):
            role = "software"
        ambiguous, reason = _ambiguity(text, start, end, "brand")
        if re.search(r"\bwithout\s+(?:an?\s+)?$", _clause_prefix(text, start), re.I):
            ambiguous, reason = True, "brand of an omitted accessory does not identify the manufacturer"
        found.append(_record(text, start, end, kind="brand", value=brand, unit=None,
                             operator=op, explicit=explicit,
                             role=role, coverage=compatibility,
                             ambiguous=ambiguous, uncertainty=reason))
    return sorted(found, key=lambda row: (row["span"][0], row["kind"]))


def _comparable(q: dict[str, Any], p: dict[str, Any], accessory_query: bool) -> bool:
    if q["kind"] != p["kind"]:
        return False
    kind = q["kind"]
    if kind == "brand":
        query_role = "compatibility" if accessory_query else q["role"]
        return query_role == p["role"]
    if kind in {"model", "storage", "length"}:
        return q["role"] == p["role"]
    # Unlabelled wattage is not evidence about an explicitly labelled input/output.
    return q["role"] == p["role"] or {q["role"], p["role"]} == {"rated", "output"}


def _numeric_relation(q: dict[str, Any], p: dict[str, Any]) -> tuple[str, str]:
    if q["operator"] == "neq" or p["operator"] == "neq":
        equal = math.isclose(q["value"], p["value"], rel_tol=1e-9, abs_tol=1e-8)
        if not equal:
            return ("match", "explicit excluded value differs") if q["operator"] == "neq" and p["operator"] == "eq" else ("unknown", "negation does not identify a value")
        if q["operator"] == p["operator"] == "neq":
            return "match", "both explicitly exclude the same value"
        return "conflict", "value is explicitly excluded"
    ql, qu = q["lower"], q["upper"]
    pl, pu = p["lower"], p["upper"]
    ql, qu = (-math.inf if ql is None else ql), (math.inf if qu is None else qu)
    pl, pu = (-math.inf if pl is None else pl), (math.inf if pu is None else pu)
    # For explicitly supported intervals the product offers each value in the range.
    if p["coverage"] and p["operator"] == "range":
        if pl <= ql and qu <= pu:
            return "match", "query lies in an explicitly supported range"
        if qu < pl or ql > pu:
            return "conflict", "query lies outside the explicitly supported range"
        return "unknown", "supported range only partly covers the query"
    if q["operator"] == p["operator"] == "eq":
        if math.isclose(ql, pl, rel_tol=1e-9, abs_tol=1e-7):
            return "match", "equal canonical values"
        relative = abs(ql - pl) / max(abs(ql), abs(pl), 1e-12)
        if q["kind"] == "storage" and relative <= 0.10:
            return "unknown", "nearby capacity may reflect binary/decimal conventions or rounding"
        if q["kind"] == "length" and relative <= 0.03 and not q["explicit"]:
            return "unknown", "nearby dimension may reflect nominal rounding"
        if q["kind"] == "power" and not q["explicit"]:
            return "unknown", "nominal wattages alone do not establish compatibility"
        return "conflict", "different explicit values in the same semantic role"
    overlap = max(ql, pl) <= min(qu, pu)
    if not overlap:
        return "conflict", "disjoint requirement and specification intervals"
    if q["operator"] == "gt" and pu <= ql or q["operator"] == "lt" and pl >= qu:
        return "conflict", "strict requirement boundary is not met"
    if p["operator"] in {"gt", "lt"} and q["operator"] == "eq" and ql in {pl, pu}:
        return "conflict", "product explicitly excludes the boundary value"
    lower_inside = pl > ql or (pl == ql and (q["operator"] != "gt" or p["operator"] == "gt"))
    upper_inside = pu < qu or (pu == qu and (q["operator"] != "lt" or p["operator"] == "lt"))
    if lower_inside and upper_inside:
        return "match", "entire product interval satisfies the requirement"
    return "unknown", "overlapping intervals do not establish satisfaction"


def _relation(q: dict[str, Any], p: dict[str, Any]) -> tuple[str, str]:
    if q["ambiguous"] or p["ambiguous"]:
        return "unknown", q.get("uncertainty") or p.get("uncertainty") or "ambiguous specification"
    if q["kind"] in {"model", "brand"}:
        equal = q["value"] == p["value"]
        if q["operator"] == "neq" or p["operator"] == "neq":
            if equal:
                return ("match", "same explicit exclusion") if q["operator"] == p["operator"] else ("conflict", "identifier is explicitly excluded")
            return ("match", "different non-excluded identifier") if q["operator"] == "neq" and p["operator"] == "eq" else ("unknown", "negation does not identify the product")
        return ("match", "same identifier") if equal else ("conflict", "different identifiers in the same family or role")
    return _numeric_relation(q, p)


def compare_attributes(query: str, title: str, description: str = "") -> dict[str, Any]:
    """Return ``features``, stable names/vector and auditable query-wise ``evidence``.

    Product records carry ``source`` (title/description). Conflicts require title
    evidence; description-only disagreement remains unknown. Contradictory records
    yield unknown, unless a model is explicitly listed as compatible. Brand-only
    mentions are not an authenticated manufacturer field. Features are soft ranker
    inputs: callers must not treat this parser as a certified compatibility filter.
    """
    query_attributes = extract_attributes(query)
    products = [dict(item, source=source) for source, text in (("title", title), ("description", description))
                for item in extract_attributes(text)]
    accessory_query = bool(_ACCESSORY.search(query))
    evidence: list[dict[str, Any]] = []
    for q in query_attributes:
        candidates = [p for p in products if _comparable(q, p, accessory_query)]
        compared = [(p, *_relation(q, p)) for p in candidates]
        matches = [(p, why) for p, status, why in compared if status == "match"]
        conflicts = [(p, why) for p, status, why in compared if status == "conflict"]
        uncertain = [(p, why) for p, status, why in compared if status == "unknown"]
        status, reason = "unknown", "no comparable explicit specification"
        selected: list[dict[str, Any]] = candidates
        if q["ambiguous"]:
            reason = q["uncertainty"]
        elif matches and conflicts and not (q["kind"] == "model" and all(p["coverage"] for p, _ in matches + conflicts)):
            reason = "product text contains contradictory or multiple variant specifications"
        elif matches:
            status, reason = "match", matches[0][1]
            selected = [p for p, _ in matches]
        elif conflicts and not uncertain and any(p["source"] == "title" for p, _ in conflicts):
            status, reason = "conflict", conflicts[0][1]
            selected = [p for p, _ in conflicts]
        elif uncertain:
            reason = uncertain[0][1]
        elif conflicts:
            reason = "description-only disagreement is insufficient for a hard conflict"
        evidence.append({"kind": q["kind"], "query": q, "status": status,
                         "reason": reason, "product": selected})
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["attr_query_count"] = float(len(query_attributes))
    features["attr_product_count"] = float(len(products))
    for row in evidence:
        features[f"attr_{row['status']}_count"] += 1.0
        features[f"attr_{row['kind']}_query_count"] += 1.0
        features[f"attr_{row['kind']}_{row['status']}_count"] += 1.0
        features["attr_ambiguous_query_count"] += float(row["query"]["ambiguous"])
    denominator = max(len(query_attributes), 1)
    for status in ("match", "conflict", "unknown"):
        features[f"attr_{status}_fraction"] = features[f"attr_{status}_count"] / denominator
    features["attr_has_conflict"] = float(features["attr_conflict_count"] > 0)
    return {"features": features, "feature_names": list(FEATURE_NAMES),
            "feature_vector": [features[name] for name in FEATURE_NAMES], "evidence": evidence}
