# Evaluation and release beyond this demonstration

The code currently provides a reproducible offline experiment and local search
application over historical English products. The procedures below distinguish
the existing manual workflow from proposed work. No traffic experiment, new
human annotation, live-stock integration or automated retraining is claimed.

## Labels and candidate retrieval: proposed study

The current ESCI/WANDS labels evaluate ranking inside supplied candidate pools.
To evaluate retrieval, first freeze a new query cohort, catalog snapshot and
availability rules. Sample queries independently of model errors; keep numeric
families and repeated users together where those identifiers exist. Record the
sampling probabilities and intended traffic population.

Pool candidates from the frozen retrievers and audit items missed by that pool
using a declared sampling procedure. Independent reviewers should see the query,
title and available specifications, with model names, scores and parser output
hidden. Collect relevance separately from attribute evidence. For evidence,
record original spans, units and scope, plus explicit match, explicit conflict,
insufficient information or ambiguous intent. Include disagreement review and a
held-out annotation sample. Agent-generated judgments are not human gold.

Measure ranking NDCG and Recall@K of known judged relevant items, alongside pool
coverage, unjudged rates and annotation uncertainty. Incomplete pooled judgments
do not establish absolute recall over every relevant product in the catalog;
unjudged items must not silently become irrelevant. New annotations used to
change rules become development data and cannot remain confirmatory holdout.

## Online comparison: proposed design

Randomize eligible users persistently between lexical and attribute ranking;
repeated queries from one user remain in one arm. Keep retrieval candidates,
stock/eligibility filters and catalog refreshes the same in both arms so the
comparison concerns ranking. Log assignment, displayed items, model/index
versions and exposure time before outcomes, subject to the service's privacy
policy. Exclude bots and employee traffic with rules fixed before analysis.

One proposed primary outcome is the fraction of search-using users who add a
displayed product to cart within a predeclared attribution window. Define the
window, denominator, multiple-search attribution and handling of missing events
before starting; report completed orders as a separate downstream outcome.
Guardrails should include request error rate, end-to-end p95 latency, empty
results, unavailable-item exposure, cancellation and purchases of mismatched
items where reliable labels exist. Offline NDCG is a diagnostic, not a substitute
for these outcomes.

Use user-level analysis consistent with randomization. Estimate variance from
actual pilot traffic, choose a practically relevant minimum effect, then declare
sample size, duration, stopping rule and guardrail limits before looking at arm
differences. No traffic volume, conversion baseline, required sample size or
expected lift is available in this repository. Inspect assignment balance and
logging completeness before interpreting the test.

## Manual offline runs and rollback

The implemented `catalog-search train` command selects on development data,
refits on train+development and evaluates the pinned benchmarks. Use a fresh
run directory; existing protocol, selection and summary files are refused.
The README gives the full prepare/train/index commands. This is an on-demand
offline run, not a scheduler or an incremental update from user events. Repeating
it does not turn the same public test set into fresh validation.

Before manually promoting an artifact directory, build its catalog index, check
feature/model/index hashes, run functional API checks and retain the previous
complete artifact directory. The server loads one explicit `--artifacts` path
at startup. A manual rollback therefore restarts it with the previously verified
directory; preserve that run's encoder, models, index, metadata and report
together. Automatic promotion, canary routing and hot reload are not implemented.

Training on future production data would additionally need a versioned adapter
for new labels, a temporal availability contract, fresh untouched evaluation,
monitoring for data/feature drift and a release gate. Do not feed new snapshots
through the current pinned-source checks by weakening their checksum validation.
Choose retraining frequency from measured drift, label delay and operational
cost; a calendar interval alone is not evidence that retraining helps.
