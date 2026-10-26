# Evaluation protocol

Implemented protocol, 26 October 2026. The recorded run froze configuration,
source and split hashes before training, and sealed development selections before
test scoring. See [the frozen snapshot](../reports/protocol.json),
[sealed selection](../reports/selection.json) and [results](../reports/results.md).
Test performance was not used to change features, grouping or hyperparameters.

## Task and hypothesis

The measured task is reranking the provided, judged candidate list for a query.
The primary hypothesis is that explicit attribute features improve mean query
NDCG@10 over a strong lexical model on unseen query families. BM25/TF–IDF and
simple conflict rules are additional baselines. Compare lexical and augmented
models with the same data, learner family and selection budget. A rule-only
effect is a result to report, not conceal.

Both learned variants must receive the same source information: if the augmented
variant parses product bullet points or WANDS `product_features`, the lexical
baseline also gets their raw text. Otherwise the comparison confounds additional
information with the attribute representation. Report a source-field ablation
separately if additional fields themselves are the intended contribution.

A catalogue search demonstration may retrieve additional products. Its full
catalogue recall is not measured by the judged candidate benchmark. Unjudged
pairs are unknown, not automatically Irrelevant. These data do not measure clicks,
conversion, revenue, money saved, or temporal deployment performance.

## ESCI source and benchmark contract

- Primary release: [Amazon ESCI repository](https://github.com/amazon-science/esci-data).
  Pin the repository revision and actual LFS-object SHA256; do not replace actual
  file counts with the older paper or README's differing counts.
- Restrict the main study to `small_version == 1` and `product_locale == "us"`.
  `small_version` is the ranking task subset; the large classification subset is
  not an extra source of training labels for held-out ranking queries.
- Join product metadata on **both** `product_locale` and `product_id`, with a
  checked many-to-one relationship. Product IDs and locale/ID pairs must not be
  assumed globally interchangeable. Verify that one query ID has one query text
  and locale, and that a query does not cross the released train/test split.
- Allowed model inputs are query text and the declared product text/attributes.
  `example_id`, raw `query_id`, raw `product_id`, `source`, `split`, `small_version`,
  `large_version` and relevance labels are not predictive features. IDs remain
  available for joins, audits and grouping. Product popularity estimated from
  held-out relevance labels is forbidden.
- The current release has `train` and `test`. The original competition paper
  described train/public/private splits; do not call the current combined test a
  newly blinded competition private test. Keep its release name in reports.
- Queries were deliberately selected to be challenging, including attribute and
  negation patterns. This is not a representative random sample of Amazon
  traffic. Products can legitimately recur across query splits; quantify this
  and do not silently claim a new-product test.

### Gains and metric implementation

The primary [paper, §3.1](https://arxiv.org/html/2206.06588v1#S3.SS1) and
[official training script](https://github.com/amazon-science/esci-data/blob/main/ranking/train.py)
specify gains **E = 1, S = 0.1, C = 0.01, I = 0**. Use these directly:

`DCG@k = sum(gain(label_at_rank_r) / log2(r + 1), r=1..min(k,n))`

`NDCG@k = DCG@k / IDCG@k`, followed by an unweighted mean over queries.
Do not apply `2**gain - 1` a second time. For integer learner labels
I/C/S/E = 0/1/2/3, explicitly configure the learner's gain function when supported;
its default exponential gains are a different objective. Validate metric order,
an ideal ranking, an E/S swap, ties and all-I queries with hand-calculated tests.
If IDCG is zero, assign NDCG zero and separately report that query count; do not
drop it after seeing which method benefits. All methods use the same query set.
Score ties use one declared label-blind deterministic order, preferably a seeded
hash of query/product IDs, not incoming CSV order.

There is an upstream inconsistency: the public
[qrels helper](https://github.com/amazon-science/esci-data/blob/main/ranking/prepare_trec_eval_files.py)
maps S to 2 and C to 3, while its example trec command maps 2 to 0.01 and 3 to
0.1. That swaps S/C relative to the paper and training script. Do not copy that
mapping. Our explicitly recorded metric is authoritative for this study; no
claim of exact leaderboard reproduction follows from using the same dataset.
Full-list NDCG is a useful secondary metric because the paper's task is not
restricted to top ten. NDCG@10 is our predeclared product-facing cutoff.

## Locked split and model selection

The following deterministic policy is implemented and recorded in the frozen
configuration and source hashes.

1. Retain the released test set for one final comparison. Inspecting its schema,
   identifier overlap and unlabeled query text for leakage checks is permitted;
   inspecting model quality on it during development is not.
2. Define conservative families from query strings alone: Unicode NFKC,
   case-folding, whitespace/punctuation normalization and replacement of numeric
   literals with a placeholder. Keep remaining words, including brands and unit
   words. Do not learn synonym groups or normalize using relevance labels or our
   attribute parser. Record the exact normalizer and examples before training.
3. Remove from the training/development pool any official-train family that also
   occurs in official test. Remove training rows, not convenient test errors.
   Report dropped rows/families and preserve a separate manifest of original
   split membership. This is a stricter training protocol, not an untouched
   reproduction of published baseline scores.
4. Partition the remaining official-train families by a seeded stable hash into
   train and development (seed `20261026`, development fraction 15%).
   All candidates for a query and all members of a family stay together. Freeze
   any size cap by family hash before training; never select easy queries by
   measured score. Fit vocabulary/IDF, learned parsers, feature selectors and
   model parameters only on training; development selects the declared finite
   configuration set. Any calibration requires its own training-side families.
5. Lock the selected configurations and preprocessing hashes before opening
   official-test predictions. The final augmented-minus-lexical NDCG@10
   difference is the one primary comparison. Other comparisons/slices are
   secondary and must not retroactively become the main claim.

Use paired bootstrap by family, retaining all its queries/candidates within a
resample. Declare seed, repetition count and confidence level before evaluation.
Report effect size, interval, query/family counts and per-query differences;
confidence intervals do not fix leakage or make dataset-selected traffic
representative. Attribute slices must be determined by fixed query/metadata
rules or independent annotations, not by which model improved. A new-product or
new-category test is a separately declared stress test; constructing it may
change the data distribution and must not be called the original official test.

## WANDS external catalogue transfer

[WANDS](https://github.com/wayfair/WANDS) is released under the repository's MIT
license and provides query/product/label tables. The `*.csv` files are actually
tab-separated. `scripts/prepare_wands.py` checks keys, duplicate candidates,
foreign-key coverage and label values, and emits normalized Parquet plus a
manifest containing immutable source URLs, actual bytes/SHA256 and the complete
license notice. Raw downloads are not saved by default. Keep normalized/raw data
outside version control; publish the adapter, manifests and evaluation results.

All WANDS queries are `domain=wands, split=test`: this is **our external-test
assignment**, not an assertion that WANDS published an official train/test
partition. No WANDS relevance labels are used for fitting, choosing parsers,
hyperparameters, thresholds, or deciding which feature variant to retain.
Queries, products and judged pairs must be counted from the actual source bytes
before evaluation. Missing query/product keys are errors. Duplicate annotation
rows are resolved under the following pre-performance policy, never silently.

At source commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`, the actual TSVs
contain 480 queries, 42,994 products and 233,448 annotation rows. There are 1,467
duplicate query/product groups (1,575 extra rows), of which 14 have contradictory
labels. Before any WANDS model evaluation the project approved: collapse
same-label duplicates into one candidate; exclude exactly the 14 conflicting
pairs from the primary candidate set. Thus the main set has 231,859 unique
query/product candidates. Preserve every source label/annotation ID of ambiguous
pairs in the manifest ledger and `ambiguous_pairs.parquet`, including minimum and
maximum labels for frozen-scorer sensitivity. Never choose the favourable label.
Nonconflicting pairs retain all source annotation IDs as a list. The original
annotation counts and source hashes remain available. This is a declared cleaned
evaluation set, not an unmodified official WANDS result. The actual product file
uses `category hierarchy` with a space, which the adapter supports.

Labels stay `Exact`, `Partial`, `Irrelevant`; Partial is not automatically ESCI S
or a merge of S/C. Recommended transfer primary is Exact-vs-other NDCG@10 and
Exact Recall@K **within judged pools**, so the common semantic claim is explicit.
Declare graded sensitivity maps before evaluation, e.g. gains E/P/I = 1/0.1/0
and 1/0.5/0. These are analytical conventions, not observed purchase utilities.
Do not choose the map that makes our method win or compare absolute NDCG across
different label scales as if domains had equal difficulty. Report each frozen
model's paired transfer difference, and bootstrap WANDS by query/family rather
than treating all query-product rows as independent.

WANDS has no dedicated brand field; the adapter leaves brand empty instead of
guessing from a title. `product_features` is retained as `attributes` for an
explicit, frozen feature adapter. `query_class` is preserved as `query_category`
for diagnostics only, because its availability as an inference input is not
established. Query/product IDs have a separate domain column; use `(domain,id)`
keys if combining datasets. CSV row order is not time.

## Original attribute audit without parser-as-gold

No new human labels currently exist. A review performed by an AI agent must be
labelled `agent_review`, not “human annotation”, “expert gold” or independent
human confirmation. Official ESCI/WANDS relevance labels remain the primary
human-judged outcomes and must not be overwritten by our audit.

Before reading parser outputs or model errors, sample audit pairs by stable hash
from locked held-out families. Use a declared mixture of a random slice and
simple query-pattern strata (numbers/units, sizes/counts, model-like tokens,
negation). Those rules select cases; they do not assign correctness. Retain
selection IDs, strata and inclusion counts. Enrichment changes prevalence, so
audit percentages describe that enriched sample, not all search traffic.

The annotation view contains only the original query and product title,
description and explicit metadata. Hide parser output, score, model identity and
the ESCI relevance label during the audit. For each queried constraint record:

- the literal query span and the product evidence span;
- type and unit, or `unresolved`;
- `explicit_match`, `explicit_conflict`, `insufficient_information`,
  `ambiguous_intent`, or `not_applicable`;
- a short reason, reviewer identity/type and review timestamp.

“Not mentioned” is insufficient information, not an explicit conflict. Model
numbers, dates, dimensions, counts and pack sizes are different; one cannot
equate all numeric substrings. A larger power rating may be compatible in one
scenario and unsuitable in another. Terms such as “large”, style/colour names,
“for”, approximate sizes and ranges can be subjective or underspecified. Preserve
ambiguity rather than resolving it in favour of the parser. Normalize a unit
only when dimension and intended attribute agree. Do not infer compatibility
from a model-family substring alone.

Use roughly 100 pairs for an initial agent/manual-reading audit only if its
provenance is explicit. A later human study should add an independent reviewer
and disagreements, with a frozen untouched sample. If audit findings change
rules or features, that slice becomes development material; a separately frozen
holdout is needed for any confirmatory audit claim. A second AI agent is a
cross-check, not a replacement for independent human labels. Do not invent
unattended annotations to reach a target sample size. Report unresolved counts,
agreement where available, and illustrative failures with source IDs.

## Release boundaries

No model-quality result is asserted until frozen configuration and split hashes
are recorded. Preserve provenance, license notices, measured hardware/runtime
context and all negative results. A search interface is a usable artifact, not
evidence of relevance, money savings or scientific novelty. If attribute features
do not beat a strong lexical baseline, report that outcome and the domain/slice
where they help or fail without selecting only favourable tests.
