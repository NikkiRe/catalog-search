# Explain the project with the saved results open

Start with the concrete error: two product titles can share almost every word
while referring to different capacities, lengths or model suffixes. The project
tests whether an explicit representation of these differences helps a strong
lexical ranker. Do not describe it as an invented learning algorithm.

Be ready to walk through these decisions in code:

1. **Candidate retrieval versus ranking.** BM25 searches the demo catalog. ESCI
   and WANDS provide judged candidate pools for ranking evaluation. The saved
   NDCG does not measure recall over all products in the application.
2. **The strong reference.** The lexical model already sees character features,
   numeric token overlaps, brand text and the same descriptions/bullets. A gain
   over BM25 alone is insufficient evidence for the attribute hypothesis.
3. **Why three evidence states?** Silence in a description is not a conflicting
   specification. Attribute scope, bundles and charging wattages can be
   ambiguous. Explain a real `unknown` case rather than promising universal
   semantic understanding.
4. **Training leakage.** Entire normalized numeric query families stay together.
   Overlapping official-train families are removed before splitting train/dev;
   all official test queries remain. Product overlap is reported separately.
   The test is not a guarantee of new products or semantic family independence.
5. **The metric.** Demonstrate DCG using gains E=1, S=.1, C=.01, I=0 and logarithmic
   discount. State how ties, no-exact and all-irrelevant queries are handled.
   Show `tests/test_metrics.py`, not an unverified library default.
6. **Selection and uncertainty.** Two declared tree sizes per variant, development
   selection, then frozen test scoring. The paired interval resamples query
   families, not individual candidate rows. It measures uncertainty on this
   benchmark, not online conversion lift.
7. **The query-only ablation.** Recognizing a storage query can be useful even
   without comparing capacities. Compare that ablation before attributing an
   improvement specifically to matching product attributes.
8. **Transfer.** WANDS has Exact/Partial/Irrelevant labels and different products.
   Report its binary and graded results, including regressions. Explain the
   predeclared policy for duplicate and contradictory annotations.
9. **Engineering.** Show a query in three modes, the evidence drawer, source
   hashes, CLI, API tests and the clean installed-wheel smoke test. Scores are
   ranking margins, not calibrated confidence.

Use numbers from `reports/summary.json` and `reports/results.md`. If an interval
crosses zero, say that the experiment did not establish a positive effect. If a
slice looks better than the primary result, call it secondary. The next useful
experiment would collect fresh independent attribute judgments and measure
candidate retrieval on a complete catalog, rather than repeatedly tuning this
test set.
