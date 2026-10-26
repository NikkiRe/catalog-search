# What this project demonstrates

This maps the technical criteria in the supplied classic-ML internship description
to inspectable work. It is not an employer assessment or a claim that one project
proves every skill in that description.

- **Python and data analysis:** `data.py` downloads a pinned snapshot, validates
  checksums, joins product metadata and creates query-family partitions with
  Polars. `features.py` uses NumPy, SciPy sparse matrices and scikit-learn text
  encoders. `scripts/audit_data.py` measures missing fields, labels and candidate
  list sizes; the saved audit is in `reports/data_audit.json`.
- **SQL:** `sql/` and `scripts/sql_audit.py` independently recompute query NDCG
  from the saved candidate scores using joins, conditional gains, window ranking
  and grouped aggregation. Nine contracts check judged rows and, when prepared
  data is supplied, query partitions and product keys. The SQL result is compared
  with Python per query for all six ranking methods. This demonstrates analytical
  SQLite, not a production PostgreSQL data platform.
- **Classic ML and feature engineering:** `experiment.py` fits real LightGBM
  LambdaRank models, selects tree size/count on development data and saves text
  boosters. `features.py` supplies 28 lexical features; `attributes.py` adds 30
  typed comparison features with explicit missing/ambiguous states. Serving calls
  the saved booster's `predict` on the same feature representation. BM25, TF–IDF,
  a conflict rule and a query-only learned ablation test different explanations
  for the result.
- **Validation, metrics and statistics:** official released test queries are
  retained, overlapping training families are excluded, and vocabulary uses
  training products only. Selection is sealed before test scoring. Linear-gain
  NDCG, Exact Recall and MRR have explicit tie and zero-denominator conventions;
  the primary difference has a paired family-bootstrap interval. WANDS provides
  frozen external transfer, with separately declared label sensitivities. There
  are no timestamps supporting a temporal split in these datasets.
- **ML design and usable delivery:** the repository separates ingestion, features,
  development selection, final fitting, evaluation, retrieval and serving. The CLI,
  FastAPI service, browser interface, dependency lock, source/data/artifact hashes
  and installed-wheel checks make these stages inspectable. A tiny synthetic
  integration test executes the actual fit/select/save/evaluate lifecycle; its
  scores are not project quality results. A new output directory is required for
  each offline run.

The measured addition is small: ESCI NDCG@10 is 0.7364 versus 0.7350 for the strong
lexical ranker. The frozen WANDS evaluation did not establish an attribute benefit.
These are judged-pool ranking results, not full-catalog retrieval quality, online
conversion, revenue or proof of parser precision. See [results](../reports/results.md)
and the [evaluation protocol](evaluation_protocol.md).

The remaining internship criteria belong elsewhere. Linear/logistic regression,
forests, clustering, optimization and hands-on use of CatBoost/XGBoost need other
work or an interview demonstration; adding unused imports here would provide no
evidence. This project does not use pandas or JupyterHub. Education, full-time
student status, availability and internship duration are personal facts, not
properties of this repository. The implemented offline pipeline is not an
automatically monitored/retrained production system. The separate
[production evaluation plan](production_evaluation.md) states what is still needed.
