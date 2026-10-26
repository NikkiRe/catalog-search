# Attribute evidence contract

The parser adds thirty deterministic features to the lexical ranker. It extracts explicit storage, electrical power, physical lengths, selected model families and an intentionally limited brand vocabulary. It does not infer missing specifications or certify physical compatibility. Its evidence describes a comparison of text, not the probability that a product is relevant.

## API and status meanings

`extract_attributes(text)` returns serializable records. Each retains the original text and `[start, end)` span, normalized value/unit, operator, semantic role and ambiguity flag. Spans index the string passed to this function; if a caller cleans HTML or normalizes Unicode first, they index that cleaned string. `confidence="high"` indicates a grammar match, not a measured confidence probability. An ambiguous record must still be treated as uncertain.

`compare_attributes(query, title, description="")` returns an ordered `features` dictionary, `feature_names`, `feature_vector` and query-wise `evidence`. `FEATURE_NAMES` fixes the thirty-column order. Evidence records include the source of each product mention, either title or description.

- **match**: the compared text supports the parsed requirement in the same semantic role. This can be one agreed specification on an otherwise irrelevant product.
- **conflict**: explicit comparable text is inconsistent with the requirement. A title must support the conflict; a conflicting description by itself is insufficient. This is a soft ranking feature, not a safe automatic exclusion rule.
- **unknown**: no reliable conclusion follows. This includes missing specifications, different roles, unresolved variants/bundles, contradictory source text, incomplete model lists and unsupported syntax. Unknown is not a match or a conflict, and should not be displayed as a failed requirement.

Absence of a parsed attribute produces no evidence row. Unsupported requirements may therefore be absent rather than listed as unknown. The user interface should say “No supported attribute found” for this case. It should not imply that the query has no constraints.

## Rules and boundaries

GB/TB denote decimal bytes; GiB/TiB denote binary bytes. Storage and transfer rates, RAM and drive capacity, individual capacity and bundle totals are kept separate. Exact equal normalized values match; nearby unequal capacities remain unknown because catalog conventions and nominal rounding vary. Fractions such as `1 1/2 inch`, `1+1/4 inch` and `1½ inch` are supported; malformed fractions and nonpositive sizes abstain. Length roles distinguish width, height, length, diameter and screen size. Missing axes in a multidimensional expression are not guessed.

Numeric bounds preserve strict versus inclusive endpoints. A stated supported range such as “fits 13–15 inch laptop” differs from an uncertain measurement interval. Alternatives joined by “or” are not treated as simultaneous required values. Negated ranges are outside the current interval representation and remain unknown. The parser does not understand arbitrary Boolean expressions, conversational intentions or clauses that require general language interpretation.

`iPhone 11`, `11 Pro` and `11 Pro Max` remain distinct identifiers. Known family grammars cover iPhone, Galaxy, Pixel, selected Surface names and PlayStation; a generic alphanumeric identifier recognizer also handles formats such as C920 and WH-1000XM4. This is incomplete: Apple Watch series, MacBook generations, many Samsung names without the Galaxy prefix, CPU families, vehicle fitments and long natural-language model names are not generally resolved. Generic identifier-family equality is a textual heuristic, not a manufacturer product database. Protocols and ratings such as WiFi6, MERV8 and DDR4 are not reclassified as product models.

Brand mentions can describe the product, an accessory target or bundled software. “For Samsung” is separated from manufacturer-like mentions; Microsoft Office/Windows mentions have a software role. The manufacturer role is still a textual inference, not an authenticated manufacturer field. Software availability, product type and genuine-brand status require other evidence. In “without an Apple pencil,” the omitted accessory's brand does not constrain the product manufacturer.

Different nominal wattages do not by themselves establish charger incompatibility. A 100W USB-PD adapter may negotiate lower power; voltage profiles, connectors, cable ratings and certification are not modeled. Explicit exact/bounded textual requirements are compared as written. Watts are distinct from watt-hours, input/output from rated/equivalent power, and W/L/H/D dimension shorthand is marked ambiguous. Physical safety, device fit, electrical approvals and vendor warranties are outside the parser contract.

Count, volume, mass, apparel/shoe size, frequency, voltage, dimensions without explicit units and general materials/colors are not currently parsed. Expanding these families needs a separate contract and label-blind coverage review; every numeral is not a product constraint.

## Label-blind development audit

Before model fitting, only the released ESCI US-small **official training query text** was inspected. The audit projected query ID/text plus split/locale/subset selectors; it did not read relevance labels. This audit corpus includes official-training queries later excluded from the family-purged modeling cohort; it is not an independently held-out parser-coverage estimate. WANDS and released test examples were not used to choose parser revisions. On 20,888 distinct `(query_id, query)` official-training rows, the initial parser emitted attributes for 1,447 queries. A review found dimension-W ambiguity, Microsoft Office prose, brand alternatives, incomplete fractions and ordinary-word/model-token artifacts. Narrow guards and fraction handling were added before the parser freeze.

After those guards, 1,407 queries had at least one emitted record: 603 model, 481 length, 402 brand, 34 storage and 34 power mentions; 24 mentions were marked ambiguous. These are **emission counts**, not accuracy, recall or human-annotated coverage. Categories can overlap within a query. The parser can miss requirements or emit incorrect records outside this diagnostic sample.

Audited examples-file SHA-256: `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a`.
Parser-freeze SHA-256: `ca958699d4bf1dcf939fd5ef4ddc5eeaa967414e1925adf46b093eaf1ce7cc90`.

The lexical and augmented rankers receive the same underlying raw text. Query-type/count features can interact with other features, so an improvement of the thirty-feature block alone would not isolate the effect of agreement detection. A query-only ablation is needed for that narrower attribution. None of this audit supplies human relevance ground truth.

## Verification and the 100-case diagnostic specification

The parser's saved test module contained 42 passing regression cases at this freeze. The following separate suite contains 100 developer-authored contract cases, grouped into unit conversions (1–10), bounds (11–20), negation (21–30), models (31–40), scope/bundles (41–50), power (51–60), brands (61–70), fractions (71–80), source evidence (81–90) and abstention (91–100). Expected statuses apply only to the named `kind`, in extraction order; `[]` means there should be no record of that kind. These constructed cases test an explicit policy. Passing them is not a benchmark estimate, a blind audit or human gold. Real-catalog precision and requirement recall remain unmeasured.

Run from the repository root after installing the package:

```python
import json
import re
from pathlib import Path
from catalog_search.attributes import compare_attributes

text = Path("docs/attribute_contract.md").read_text()
block = re.search(r"```jsonl\n(.*?)\n```", text, re.S).group(1)
for line in block.splitlines():
    case = json.loads(line)
    actual = [item["status"] for item in compare_attributes(
        case["query"], case["title"], case["description"]
    )["evidence"] if item["kind"] == case["kind"]]
    assert actual == case["expected"], (case["id"], actual, case["expected"])
```

```jsonl
{"id": 1, "kind": "storage", "query": "1 TB SSD", "title": "1000 GB SSD", "description": "", "expected": ["match"]}
{"id": 2, "kind": "storage", "query": "1 TiB SSD", "title": "1024 GiB SSD", "description": "", "expected": ["match"]}
{"id": 3, "kind": "storage", "query": "500 MB drive", "title": "0.5 GB drive", "description": "", "expected": ["match"]}
{"id": 4, "kind": "storage", "query": "1 GiB drive", "title": "1024 MiB drive", "description": "", "expected": ["match"]}
{"id": 5, "kind": "length", "query": "width 1 inch", "title": "width 25.4 mm", "description": "", "expected": ["match"]}
{"id": 6, "kind": "length", "query": "6 inch long cable", "title": "15.24 cm long cable", "description": "", "expected": ["match"]}
{"id": 7, "kind": "length", "query": "length 1 m", "title": "100 cm long", "description": "", "expected": ["match"]}
{"id": 8, "kind": "power", "query": "output 0.065 kW", "title": "65 watts output", "description": "", "expected": ["match"]}
{"id": 9, "kind": "power", "query": "1 kW generator", "title": "1000 W generator", "description": "", "expected": ["match"]}
{"id": 10, "kind": "length", "query": "width 1.5 inch", "title": "width 38.1 mm", "description": "", "expected": ["match"]}
{"id": 11, "kind": "storage", "query": "at least 128 GB", "title": "256 GB", "description": "", "expected": ["match"]}
{"id": 12, "kind": "storage", "query": "at least 128 GB", "title": "64 GB", "description": "", "expected": ["conflict"]}
{"id": 13, "kind": "storage", "query": "up to 128 GB", "title": "256 GB", "description": "", "expected": ["conflict"]}
{"id": 14, "kind": "storage", "query": "under 128 GB", "title": "128 GB", "description": "", "expected": ["conflict"]}
{"id": 15, "kind": "storage", "query": "over 128 GB", "title": "128 GB", "description": "", "expected": ["conflict"]}
{"id": 16, "kind": "storage", "query": "over 128 GB", "title": "129 GB", "description": "", "expected": ["match"]}
{"id": 17, "kind": "length", "query": "13-15 inch laptop", "title": "14 inch laptop", "description": "", "expected": ["match"]}
{"id": 18, "kind": "length", "query": "13-15 inch laptop", "title": "16 inch laptop", "description": "", "expected": ["conflict"]}
{"id": 19, "kind": "length", "query": "width 1 inch to 3 cm", "title": "width 2.75 cm", "description": "", "expected": ["match"]}
{"id": 20, "kind": "length", "query": "width 3-1 cm", "title": "width 2 cm", "description": "", "expected": ["unknown"]}
{"id": 21, "kind": "storage", "query": "not 128 GB", "title": "128 GB", "description": "", "expected": ["conflict"]}
{"id": 22, "kind": "storage", "query": "not 128 GB", "title": "256 GB", "description": "", "expected": ["match"]}
{"id": 23, "kind": "model", "query": "not iPhone 11", "title": "iPhone 11", "description": "", "expected": ["conflict"]}
{"id": 24, "kind": "model", "query": "not iPhone 11", "title": "iPhone 12", "description": "", "expected": ["match"]}
{"id": 25, "kind": "model", "query": "iPhone 11 case", "title": "not for iPhone 11", "description": "", "expected": ["conflict"]}
{"id": 26, "kind": "model", "query": "iPhone 11 case", "title": "case", "description": "not for iPhone 11", "expected": ["unknown"]}
{"id": 27, "kind": "brand", "query": "not Samsung", "title": "Samsung phone", "description": "", "expected": ["conflict"]}
{"id": 28, "kind": "storage", "query": "without 128 GB", "title": "128 GB", "description": "", "expected": ["conflict"]}
{"id": 29, "kind": "length", "query": "not 13-15 inch laptop", "title": "14 inch laptop", "description": "", "expected": ["unknown"]}
{"id": 30, "kind": "storage", "query": "not 128 GB but 256 GB", "title": "256 GB", "description": "", "expected": ["match", "match"]}
{"id": 31, "kind": "model", "query": "iPhone 11 case", "title": "iPhone 11 Pro case", "description": "", "expected": ["conflict"]}
{"id": 32, "kind": "model", "query": "iPhone 11 Pro", "title": "iPhone 11 Pro Max", "description": "", "expected": ["conflict"]}
{"id": 33, "kind": "model", "query": "iPhone SE", "title": "iPhone SE", "description": "", "expected": ["match"]}
{"id": 34, "kind": "model", "query": "Galaxy S21", "title": "Galaxy S21 Ultra", "description": "", "expected": ["conflict"]}
{"id": 35, "kind": "model", "query": "Pixel 6a", "title": "Pixel 6", "description": "", "expected": ["conflict"]}
{"id": 36, "kind": "model", "query": "PS5 controller", "title": "PlayStation 5 controller", "description": "", "expected": ["match"]}
{"id": 37, "kind": "model", "query": "WH1000XM4 headphones", "title": "WH-1000XM4 headphones", "description": "", "expected": ["match"]}
{"id": 38, "kind": "model", "query": "Logitech C920", "title": "Logitech C920s", "description": "", "expected": ["conflict"]}
{"id": 39, "kind": "model", "query": "iPhone 12 case", "title": "case for iPhone 11/12", "description": "", "expected": ["unknown"]}
{"id": 40, "kind": "model", "query": "Lenovo Thinkpad T480", "title": "Lenovo T490", "description": "", "expected": ["conflict"]}
{"id": 41, "kind": "storage", "query": "16 GB RAM", "title": "16 GB storage", "description": "", "expected": ["unknown"]}
{"id": 42, "kind": "storage", "query": "128 GB drive", "title": "128 GB RAM", "description": "", "expected": ["unknown"]}
{"id": 43, "kind": "storage", "query": "500 MB storage", "title": "500 MB/s drive", "description": "", "expected": ["unknown"]}
{"id": 44, "kind": "length", "query": "width 10 cm", "title": "height 10 cm", "description": "", "expected": ["unknown"]}
{"id": 45, "kind": "length", "query": "15 inch screen", "title": "15 inch cable", "description": "", "expected": ["unknown"]}
{"id": 46, "kind": "storage", "query": "128 GB drive", "title": "2 x 64 GB drives", "description": "", "expected": ["unknown"]}
{"id": 47, "kind": "storage", "query": "64 GB drive", "title": "64 GB / 128 GB variants", "description": "", "expected": ["unknown"]}
{"id": 48, "kind": "storage", "query": "128 GB drive", "title": "128 GB total", "description": "", "expected": ["unknown"]}
{"id": 49, "kind": "storage", "query": "128 GB or 256 GB", "title": "256 GB", "description": "", "expected": ["unknown", "unknown"]}
{"id": 50, "kind": "storage", "query": "16 GB RAM", "title": "16 GB RAM and 256 GB SSD", "description": "", "expected": ["match"]}
{"id": 51, "kind": "power", "query": "65W charger", "title": "100W charger", "description": "", "expected": ["unknown"]}
{"id": 52, "kind": "power", "query": "exactly 65W charger", "title": "100W charger", "description": "", "expected": ["conflict"]}
{"id": 53, "kind": "power", "query": "at least 65W charger", "title": "100W charger", "description": "", "expected": ["match"]}
{"id": 54, "kind": "power", "query": "60W bulb", "title": "8W LED, 60W equivalent", "description": "", "expected": ["unknown"]}
{"id": 55, "kind": "power", "query": "65W output", "title": "65W input", "description": "", "expected": ["unknown"]}
{"id": 56, "kind": "power", "query": "65W charger", "title": "65W output", "description": "", "expected": ["match"]}
{"id": 57, "kind": "power", "query": "65W charger", "title": "65Wh battery", "description": "", "expected": ["unknown"]}
{"id": 58, "kind": "power", "query": "30w 48l folding table", "title": "30W lamp", "description": "", "expected": ["unknown"]}
{"id": 59, "kind": "power", "query": "80w x 60l tapestry", "title": "80W heater", "description": "", "expected": ["unknown"]}
{"id": 60, "kind": "power", "query": "65W adapter", "title": "supports 45-100W output", "description": "", "expected": ["match"]}
{"id": 61, "kind": "brand", "query": "Samsung phone", "title": "Sony phone", "description": "", "expected": ["conflict"]}
{"id": 62, "kind": "brand", "query": "Samsung charger", "title": "Anker charger for Samsung", "description": "", "expected": ["match"]}
{"id": 63, "kind": "brand", "query": "Samsung phone", "title": "case for Samsung", "description": "", "expected": ["unknown"]}
{"id": 64, "kind": "brand", "query": "Lenovo or Dell laptop", "title": "Lenovo laptop", "description": "", "expected": ["unknown", "unknown"]}
{"id": 65, "kind": "brand", "query": "laptop with Microsoft Office", "title": "Dell laptop", "description": "", "expected": ["unknown"]}
{"id": 66, "kind": "brand", "query": "Microsoft Office", "title": "Microsoft Office", "description": "", "expected": ["match"]}
{"id": 67, "kind": "brand", "query": "apple slicer", "title": "red apple slices", "description": "", "expected": []}
{"id": 68, "kind": "brand", "query": "Apple iPhone 11", "title": "Apple iPhone 11", "description": "", "expected": ["match"]}
{"id": 69, "kind": "brand", "query": "Google Chromecast", "title": "Google Nest", "description": "", "expected": ["match"]}
{"id": 70, "kind": "brand", "query": "case without an Apple pencil", "title": "Apple case", "description": "", "expected": ["unknown"]}
{"id": 71, "kind": "length", "query": "1/2 inch socket", "title": "12.7 mm socket", "description": "", "expected": ["match"]}
{"id": 72, "kind": "length", "query": "1½ inch drain", "title": "38.1 mm drain", "description": "", "expected": ["match"]}
{"id": 73, "kind": "length", "query": "1 and 1/2 inch wide belt", "title": "width 38.1 mm belt", "description": "", "expected": ["match"]}
{"id": 74, "kind": "length", "query": "¾ inch drain", "title": "19.05 mm drain", "description": "", "expected": ["match"]}
{"id": 75, "kind": "length", "query": "⅝ inch drain", "title": "15.875 mm drain", "description": "", "expected": ["match"]}
{"id": 76, "kind": "length", "query": "1+1/4 inch drain", "title": "31.75 mm drain", "description": "", "expected": ["match"]}
{"id": 77, "kind": "length", "query": "1 +/4in drain", "title": "4in drain", "description": "", "expected": ["unknown"]}
{"id": 78, "kind": "length", "query": "1/0 inch", "title": "1 inch", "description": "", "expected": []}
{"id": 79, "kind": "length", "query": "00m bracket", "title": "1m bracket", "description": "", "expected": ["unknown"]}
{"id": 80, "kind": "length", "query": "1/2 to 1 inch drain", "title": "19.05 mm drain", "description": "", "expected": ["match"]}
{"id": 81, "kind": "storage", "query": "128 GB SSD", "title": "SSD", "description": "Storage: 128 GB", "expected": ["match"]}
{"id": 82, "kind": "storage", "query": "128 GB SSD", "title": "SSD", "description": "Storage: 256 GB", "expected": ["unknown"]}
{"id": 83, "kind": "storage", "query": "128 GB SSD", "title": "256 GB SSD", "description": "Storage: 128 GB", "expected": ["unknown"]}
{"id": 84, "kind": "storage", "query": "128 GB SSD", "title": "128 GB SSD", "description": "Storage: 256 GB", "expected": ["unknown"]}
{"id": 85, "kind": "model", "query": "iPhone 11 case", "title": "case", "description": "for iPhone 11", "expected": ["match"]}
{"id": 86, "kind": "model", "query": "iPhone 11 case", "title": "case", "description": "for iPhone 12", "expected": ["unknown"]}
{"id": 87, "kind": "model", "query": "iPhone 11 case", "title": "fits iPhone 11 and iPhone 12", "description": "", "expected": ["match"]}
{"id": 88, "kind": "storage", "query": "64 GB drive", "title": "64 GB drive plus 16 GB RAM", "description": "", "expected": ["match"]}
{"id": 89, "kind": "power", "query": "over 65W", "title": "65-100W", "description": "", "expected": ["unknown"]}
{"id": 90, "kind": "power", "query": "under 65W", "title": "45-65W", "description": "", "expected": ["unknown"]}
{"id": 91, "kind": "model", "query": "book 1984", "title": "plain book", "description": "", "expected": []}
{"id": 92, "kind": "model", "query": "MERV8", "title": "MERV13", "description": "", "expected": []}
{"id": 93, "kind": "model", "query": "WiFi6", "title": "WiFi5", "description": "", "expected": []}
{"id": 94, "kind": "model", "query": "binder4", "title": "binder5", "description": "", "expected": []}
{"id": 95, "kind": "model", "query": "model binder4", "title": "model binder5", "description": "", "expected": ["conflict"]}
{"id": 96, "kind": "length", "query": "2 in 1 convertible", "title": "convertible", "description": "", "expected": []}
{"id": 97, "kind": "storage", "query": "128 Gb", "title": "128 GB", "description": "", "expected": ["unknown"]}
{"id": 98, "kind": "storage", "query": "1 TB SSD", "title": "1024 GB SSD", "description": "", "expected": ["unknown"]}
{"id": 99, "kind": "length", "query": "10m cable", "title": "shoe size 10 M", "description": "", "expected": ["unknown"]}
{"id": 100, "kind": "model", "query": "iPhone 12", "title": "iPhone 11 or iPhone 12", "description": "", "expected": ["unknown"]}
```

## Primary references

The unit policy follows [NIST SI prefixes](https://www.nist.gov/pml/owm/metric-si-prefixes), [NIST's account of IEC binary prefixes](https://physics.nist.gov/cuu/Units/binary.html) and [NIST length units](https://www.nist.gov/pml/owm/si-units-length). The conservative charger policy follows the negotiated-power description in the [USB-IF USB Power Delivery overview](https://www.usb.org/usb-charger-pd). These references justify unit and protocol distinctions; they do not validate catalog text or establish compatibility of any product in this dataset.

The [official ESCI release](https://github.com/amazon-science/esci-data) supplies query/product relevance judgements. Those labels evaluate the ranker; they are not specification-level annotations for this parser. See the separate [evaluation protocol](evaluation_protocol.md) for grouped model comparisons and external evaluation.
