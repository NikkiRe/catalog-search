#!/usr/bin/env python3
"""Build figures and a results document from one completed frozen experiment.

No fitting or prediction occurs here. Error examples are deterministic post-hoc
illustrations selected from saved scores, and never become training examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "catalog-search-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from catalog_search.attributes import compare_attributes
from catalog_search.data import load_pairs
from catalog_search.text import document, stable_hash


ORDER = ("tfidf", "bm25", "rule", "baseline", "no_product_evidence", "attribute")
NAMES = {"tfidf": "TF–IDF", "bm25": "BM25", "rule": "Lexical + conflict rule",
         "baseline": "Lexical LambdaMART", "no_product_evidence": "Lexical + query attributes",
         "attribute": "Lexical + attribute evidence"}
COLORS = {"attribute": "#087f8c", "baseline": "#253858", "no_product_evidence": "#687ea2",
          "rule": "#c77743", "bm25": "#a6b1bf", "tfidf": "#c3cad3"}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return value is not None and math.isfinite(float(value))


def number(value, places=4):
    return f"{float(value):.{places}f}" if finite(value) else "n/a"


def signed(value, places=4):
    return f"{float(value):+.{places}f}" if finite(value) else "n/a"


def relative(candidate, baseline):
    return 100 * (float(candidate) / float(baseline) - 1) if finite(candidate) and finite(baseline) and baseline else None


def relative_text(candidate, baseline):
    value = relative(candidate, baseline)
    return f"{value:+.2f}%" if value is not None else "undefined (zero baseline)"


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(escape(v) for v in row) + " |" for row in rows])


def modes(models):
    return [name for name in ORDER if name in models] + sorted(set(models) - set(ORDER))


def label(name):
    return NAMES.get(name, name)


def interval(comparison):
    return f"{signed(comparison['difference'], 6)} [{signed(comparison['ci_low'], 6)}, {signed(comparison['ci_high'], 6)}]"


def interpretation(comparison):
    if comparison["ci_low"] > 0:
        return "The paired interval lies above zero: the primary comparison favors attribute evidence on this evaluated cohort."
    if comparison["ci_high"] < 0:
        return "The paired interval lies below zero: the primary comparison favors the lexical baseline on this evaluated cohort."
    return "The paired interval includes zero; the primary comparison does not establish a directional improvement on this evaluated cohort."


def validate(summary, per_query):
    if summary.get("status") != "completed":
        raise ValueError("Report requires a completed experiment summary")
    models = summary["esci"]
    if not {"attribute", "baseline"}.issubset(models):
        raise ValueError("Primary attribute/baseline results are missing")
    for mode in modes(models):
        actual = per_query[f"{mode}_ndcg10"].mean()
        if not math.isclose(actual, models[mode]["ndcg10"], abs_tol=1e-10):
            raise ValueError(f"Summary and per-query results disagree for {mode}")
    expected = models["attribute"]["ndcg10"] - models["baseline"]["ndcg10"]
    comparison = summary["primary_difference"]
    if not math.isclose(expected, comparison["difference"], abs_tol=1e-10) or per_query.height != comparison["queries"]:
        raise ValueError("Primary comparison does not match the supplied per-query cohort")
    if comparison["ci_low"] > comparison["ci_high"]:
        raise ValueError("Invalid primary confidence interval")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#b8c0cc", "axes.labelcolor": "#243247",
        "text.color": "#243247", "xtick.color": "#46536a", "ytick.color": "#46536a",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none"})


def bar_plot(ax, models, title):
    names = modes(models)
    values = [models[name]["ndcg10"] for name in names]
    bars = ax.barh(np.arange(len(names)), values, height=.62,
                   color=[COLORS.get(name, "#a6b1bf") for name in names])
    ax.set_yticks(np.arange(len(names)), [label(name) for name in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_xlabel("Mean query NDCG@10 (0–1; higher is better)")
    ax.set_title(title, loc="left", pad=14)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e6eaf0", linewidth=.8)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        ax.text(min(value + .014, .985), bar.get_y() + bar.get_height()/2,
                f"{value:.4f}", va="center", ha="left" if value < .91 else "right",
                color="#243247", fontsize=10, fontweight="bold")


def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for extension in ("png", "svg"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        result[extension] = path
    plt.close(fig)
    return result


def figures(summary, directory):
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.8, 1]})
    bar_plot(left, summary["esci"], "ESCI · released US-small test")
    comparison = summary["primary_difference"]
    delta, lo, hi = (comparison[key] for key in ("difference", "ci_low", "ci_high"))
    extent = max(abs(delta), abs(lo), abs(hi), .001) * 1.35
    color = "#087f8c" if delta >= 0 else "#b65742"
    right.axvline(0, color="#6e7788", linewidth=1, linestyle="--")
    right.plot([lo, hi], [0, 0], color=color, linewidth=3, solid_capstyle="round")
    right.scatter([delta], [0], s=65, color=color, zorder=3)
    right.plot([lo, lo], [-.04, .04], color=color)
    right.plot([hi, hi], [-.04, .04], color=color)
    right.set_xlim(-extent, extent)
    right.set_ylim(-.4, .5)
    right.set_yticks([])
    right.set_title("Predeclared primary comparison", loc="left", pad=14)
    right.set_xlabel("Δ mean query NDCG@10\nAttribute evidence − lexical baseline")
    right.text(0, .25, f"{signed(delta, 6)}\n95% CI [{signed(lo, 6)}, {signed(hi, 6)}]", ha="center", fontsize=11)
    right.text(.5, .03, "Paired query-family bootstrap", transform=right.transAxes, ha="center", fontsize=9)
    right.spines["left"].set_visible(False)
    fig.subplots_adjust(wspace=.6, bottom=.23)
    fig.text(.01, .02, f"Frozen judged candidate pools · {comparison['queries']:,} test queries / {comparison['families']:,} normalized families.\n"
             "Bars are point estimates. The interval resamples families; it does not estimate live-traffic or full-catalogue retrieval performance.", fontsize=8.5, color="#5d687a")
    output = {"esci": save_figure(fig, directory, "esci_results")}
    wands = summary.get("wands", {})
    scenarios = wands.get("scenarios", {})
    main = scenarios.get("consensus_main", {})
    if "exact_binary" in main:
        row = main["exact_binary"]
        fig, ax = plt.subplots(figsize=(8.6, 4.9))
        bar_plot(ax, row["models"], "WANDS · frozen external transfer")
        fig.subplots_adjust(left=.3, bottom=.2)
        fig.text(.01, .02, f"Consensus candidate pairs · {row['queries']:,} queries / {row['pairs']:,} pairs. Exact=1, Partial=0, Irrelevant=0.\n"
                 "All models and vocabulary remain fitted on ESCI. Point estimates; alternate gains and ambiguous labels are reported separately.", fontsize=8.5, color="#5d687a")
        output["wands"] = save_figure(fig, directory, "wands_transfer")
    return output


def error_analysis(summary, per_query, scores, data_dir, seed):
    # Analysis is opened only after final evaluation; no model code runs here.
    prepared = load_pairs(data_dir, "test")
    expected_keys = set(map(tuple, scores.select("query_id", "product_id").iter_rows()))
    actual_keys = set(map(tuple, prepared.select("query_id", "product_id").iter_rows()))
    if expected_keys != actual_keys:
        raise ValueError("Scores and data-dir refer to different test candidate pools")
    merged = scores.join(prepared.select("query_id", "product_id", "query", "title", "brand", "description", "attributes"),
                         on=["query_id", "product_id"], how="left", validate="1:1")
    rows = per_query.with_columns((pl.col("attribute_ndcg10") - pl.col("baseline_ndcg10")).alias("delta")).to_dicts()
    for row in rows:
        row["selection_tie_hash"] = stable_hash(str(row["query_id"]), seed)
    choices = {"largest_gains": sorted(rows, key=lambda r: (-r["delta"], r["selection_tie_hash"]))[:5],
               "largest_losses": sorted(rows, key=lambda r: (r["delta"], r["selection_tie_hash"]))[:5]}
    analyses = {}
    for group_name, queries in choices.items():
        analyses[group_name] = []
        for entry in queries:
            frame = merged.filter(pl.col("query_id") == entry["query_id"])
            query = frame["query"][0]
            entry["query"] = query
            entry["top_products"] = {}
            for mode in ("baseline", "attribute"):
                selected = frame.sort([mode, "tie_order"], descending=[True, False]).head(3)
                items = []
                for product in selected.iter_rows(named=True):
                    title, brand, rest = document(product["title"], product["brand"], product["description"], product["attributes"])
                    evidence = compare_attributes(query, title, " ".join((title, brand, rest)))["evidence"]
                    items.append({"product_id": product["product_id"], "title": title, "label": product["label"],
                                  "score": product[mode], "tie_order": product["tie_order"], "attribute_evidence": evidence})
                entry["top_products"][mode] = items
            analyses[group_name].append(entry)
    return {"status": "post_hoc_illustrations_only", "selection": "Five largest and five smallest attribute-minus-baseline query NDCG@10 differences; ties by seeded query-ID hash",
            "seed": seed, "limitations": "Outcome-selected examples; not a random audit, causal explanations, attribute accuracy labels or new human annotations. No subsequent training uses these examples.",
            "examples": analyses}


def results_document(summary, analysis, image_paths, output, summary_hash, protocol=None, audit=None):
    models, comparison = summary["esci"], summary["primary_difference"]
    baseline, candidate = models["baseline"]["ndcg10"], models["attribute"]["ndcg10"]
    rows = [[label(mode), number(models[mode]["ndcg10"]), number(models[mode]["ndcg_all"]),
             number(models[mode].get("exact_recall10")), number(models[mode]["mrr"]),
             signed(models[mode]["ndcg10"] - baseline)] for mode in modes(models)]
    parts = ["# Frozen ranking results", "", f"Completed {summary['completed_at_utc']}. These results rerank the released judged candidate pools; they do not measure full-catalogue recall, conversion or business profit.", "",
        f"The predeclared primary comparison is attribute evidence versus lexical LambdaMART on every released ESCI US-small test query. Mean query NDCG@10 is **{number(candidate)} versus {number(baseline)}**: an absolute difference of **{signed(comparison['difference'], 6)}**, or **{relative_text(candidate, baseline)}** relative to the baseline. The 95% paired normalized-query-family interval is **[{signed(comparison['ci_low'], 6)}, {signed(comparison['ci_high'], 6)}]**. {interpretation(comparison)}", "",
        f"The interval uses {comparison['bootstrap_repeats']:,} seeded percentile bootstrap replicates over {comparison['families']:,} normalized families ({comparison['queries']:,} queries), retaining all queries in each resampled family. It is conditional on this released cohort, its judgments and the heuristic grouping; it does not establish robustness across time or representative user traffic.", "",
        "## ESCI comparison", "", table(["Model", "NDCG@10", "NDCG all", "Exact Recall@10", "Exact MRR", "Δ NDCG@10 vs lexical"], rows), "",
        "All scores are query means. Exact Recall@10 averages only queries with an Exact item; MRR is zero when none exists. Zero-IDCG queries have NDCG zero. The conflict-rule comparator demotes parsed conflicts below other candidates while preserving lexical order; it is a deliberately simple alternative to learning from soft evidence.", ""]
    relative_image = os.path.relpath(image_paths["esci"]["png"], output.parent)
    parts += [f"![ESCI point estimates and paired primary difference]({relative_image})", ""]
    if "no_product_evidence" in models:
        qonly = models["no_product_evidence"]["ndcg10"]
        parts += ["## Query-only attribute diagnostic", "",
            f"The model with lexical features and query-only attribute counts reaches NDCG@10 **{number(qonly)}**. The full attribute model differs by **{signed(candidate-qonly)}**; the query-only model differs from lexical by **{signed(qonly-baseline)}**. Each variant received the same development search budget. These diagnostic point comparisons separate some query-type signal from product-evidence signal; they do not measure extraction precision or identify a causal mechanism.", ""]
    parts += ["## Predeclared descriptive slices", ""]
    slice_rows = []
    for name, row in summary.get("slices", {}).items():
        metrics = row.get("metrics", {})
        slice_rows.append([name, f"{row['queries']:,}", f"{row['families']:,}"] + [number(metrics.get(mode, {}).get("ndcg10")) for mode in modes(models)])
    parts += [table(["Slice", "Queries", "Families"] + [label(mode) for mode in modes(models)], slice_rows), "",
        "These slices were specified before evaluation. They are descriptive, may overlap and have no multiple-comparison significance claim. “New to final fit” concerns all products in a query's candidate pool, not temporal cold start. Attribute-mentioned means a parser record was emitted, including ambiguous records; it is not a verified-constraint label.", ""]
    if audit:
        missing = audit["empty_field_fraction"]
        train_pool, test_pool = (audit["partitions"][part]["candidates_per_query"] for part in ("train", "test"))
        parts += ["## Descriptive metadata audit", "",
            f"Among {audit['products']:,} prepared products, {100*missing['description']:.3f}% have an empty description, "
            f"{100*missing['brand']:.3f}% an empty brand and {100*missing['attributes']:.3f}% empty bullet/attribute text. "
            "These describe the fixed shortened metadata, not parser accuracy. Missing information supports an unknown status rather than a contradiction.", "",
            f"Training judged pools have median {number(train_pool['median'], 0)} candidates, 90th percentile {number(train_pool['p90'], 0)} and maximum {train_pool['max']}; "
            f"the test maximum is {test_pool['max']}. Pool-size variation and missing specifications are part of this evaluation, not additional controlled treatments.", ""]
    wands = summary.get("wands", {})
    if wands.get("scenarios"):
        parts += ["## Frozen WANDS external evaluation", "",
            "No vocabulary, ranker, parameters or thresholds were fitted or selected on WANDS. The primary external relevance map is Exact=1, Partial=0, Irrelevant=0. WANDS Partial is not mapped to ESCI Substitute. Alternative Partial gains and ambiguous annotation scenarios remain separate descriptive evaluations.", ""]
        for scenario, maps in wands["scenarios"].items():
            map_names = list(maps)
            first = maps[map_names[0]]
            parts += [f"### {scenario}", "", f"{first['queries']:,} queries, {first['families']:,} normalized families, {first['pairs']:,} judged pairs.", ""]
            parts += [table(["Model"] + [name for name in map_names], [[label(mode)] + [number(maps[name]["models"][mode]["ndcg10"]) for name in map_names] for mode in modes(first["models"])]), ""]
            map_rows = []
            for map_name, row in maps.items():
                m = row["models"]
                map_rows.append([map_name, json.dumps(row["gain_map"], sort_keys=True), interval(row["primary_difference"]),
                                 relative_text(m["attribute"]["ndcg10"], m["baseline"]["ndcg10"]), row["zero_idcg_queries"]])
            parts += [table(["Gain map", "Values", "Attribute − lexical [95% CI]", "Relative difference", "Zero-IDCG queries"], map_rows), ""]
        parts += [wands.get("ambiguity_note", ""), "Conflicting annotations are excluded from consensus main and retained under predetermined min/max label scenarios. These scenarios are not adjudicated truth or sharp bounds on NDCG. Intervals here are descriptive external comparisons, not additional corrected confirmatory tests.", ""]
        if "wands" in image_paths:
            parts += [f"![WANDS Exact-binary transfer point estimates]({os.path.relpath(image_paths['wands']['png'], output.parent)})", ""]
    parts += ["## Deterministic post-hoc examples", "",
        "The following outcome-selected illustrations show five largest gains and five largest losses in query NDCG@10. Ties use a seeded query-ID hash. Saved JSON contains the top three baseline and attribute-ranked product IDs, original relevance labels, scores and parser text evidence. These examples are not a random error-rate audit, causal explanations, new attribute truth or new human annotations. No subsequent fitting uses them.", ""]
    for group, examples in analysis["examples"].items():
        parts += [f"### {group}", "", table(["Query", "ID", "Lexical", "Attribute", "Δ"],
                    [[row["query"], row["query_id"], number(row["baseline_ndcg10"]), number(row["attribute_ndcg10"]), signed(row["delta"])] for row in examples]), ""]
    parts += ["See [error_analysis.json](error_analysis.json) for selected products and evidence.", "",
        "## Splits, gains and reproducibility", "", table(["Partition", "Judged pairs", "Queries", "Normalized families"],
            [[row["partition"], f"{row['pairs']:,}", f"{row['queries']:,}", f"{row['families']:,}"] for row in summary["splits"]]), "",
        f"ESCI gains: `{json.dumps(summary['gains'], sort_keys=True)}`. The encoder uses unique TRAIN products only and stays frozen during the train+development final ranker refits. Each variant selects leaves/iteration count using development NDCG@10 before final test evaluation. The official test is public; the local protocol and selection were sealed before its quality was scored.", "",
        f"The preparation excluded {summary['dataset']['excluded_official_train_queries']:,} released-training queries in {summary['dataset']['excluded_families']:,} normalized families overlapping released test. Family normalization replaces numeric expressions and normalizes case/punctuation; it is a heuristic leakage guard, not a semantic near-duplicate detector. Train/test product overlap remains part of this benchmark. Neither split is a realistic time-based deployment evaluation.", "",
        f"Total recorded experiment wall time: **{summary['runtime_seconds']/60:.2f} minutes**. This includes feature generation, model selection/refits and evaluation; it is not a model-only training time. Per-phase feature timing follows.", "",
        table(["Phase", "Rows", "Columns", "Feature wall seconds"], [[name, f"{row['rows']:,}", row['features'], number(row['seconds'], 2)] for name, row in summary['feature_compute'].items()]), "",
        "The experiment summary does not contain measured peak memory or hardware power/energy consumption. No model-only memory figure is inferred from process RSS. A separate small TRAIN-only engineering microbenchmark is not an end-to-end serving benchmark.", "",
        f"Source summary SHA-256: `{summary_hash}`. Protocol: `{summary['provenance']['protocol_sha256']}`. Selection: `{summary['provenance']['selection_sha256']}`. Config: `{summary['provenance']['config_sha256']}`.", "",
        "## Scope limits", ""]
    parts += [f"- {limit}." for limit in summary.get("limitations", [])]
    if protocol:
        hardware = protocol.get("hardware", {})
        parts += ["", "## Recorded execution environment", "",
                  f"Platform: `{hardware.get('platform', 'not recorded')}`; architecture: `{hardware.get('machine', 'not recorded')}`; configured model threads: {hardware.get('threads', 'not recorded')}. This records the execution environment, not a measured hardware utilization or memory result.", "",
                  table(["Package", "Recorded version"], sorted(protocol.get("software", {}).items())), ""]
    trial_rows = []
    for mode, choice in summary["selection"]["models"].items():
        for trial in choice["trials"]:
            trial_rows.append([label(mode), trial["num_leaves"], trial["trees"], number(trial["dev_ndcg10"]), number(trial["fit_seconds"], 2),
                               "selected" if trial["num_leaves"] == choice["num_leaves"] and trial["trees"] == choice["trees"] else ""])
    parts += ["## Development selection record", "", table(["Variant", "Leaves", "Best iteration", "Dev NDCG@10", "Fit seconds", "Choice"], trial_rows), ""]
    contract = os.path.relpath(PROJECT / "docs" / "attribute_contract.md", output.parent)
    evaluation = os.path.relpath(PROJECT / "docs" / "evaluation_protocol.md", output.parent)
    parts += ["The parser's coverage is intentionally incomplete and its conflicts are soft evidence. Lexical and augmented models use identical raw text. "
              f"See [attribute contract]({contract}) and [evaluation protocol]({evaluation}).", "",
              "Dataset publishers: [Amazon ESCI](https://github.com/amazon-science/esci-data), [Wayfair WANDS](https://github.com/wayfair/WANDS).", ""]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True, help="Completed experiment reports directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="Matching prepared ESCI data directory")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports" / "results.md")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT / "docs" / "figures")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "experiment.json")
    parser.add_argument("--protocol", type=Path, help="Matching frozen protocol; defaults to report-dir/../artifacts/protocol.json")
    parser.add_argument("--data-audit", type=Path, help="Descriptive audit; defaults to report-dir/data_audit.json when present")
    args = parser.parse_args()
    summary_path = args.report_dir / "summary.json"
    summary = read_json(summary_path)
    per_query = pl.read_parquet(args.report_dir / "per_query.parquet")
    scores = pl.read_parquet(args.report_dir / "test_scores.parquet")
    validate(summary, per_query)
    config = read_json(args.config)
    if canonical_hash(config) != summary["provenance"]["config_sha256"]:
        raise ValueError("Report config does not match the frozen experiment")
    protocol_path = args.protocol or args.report_dir.parent / "artifacts" / "protocol.json"
    protocol = None
    if protocol_path.is_file():
        if digest(protocol_path) != summary["provenance"]["protocol_sha256"]:
            raise ValueError("Report protocol does not match the frozen experiment")
        protocol = read_json(protocol_path)
        for name in ("test.parquet", "products.parquet"):
            if digest(args.data_dir / name) != protocol["esci_prepared_sha256"][name]:
                raise ValueError(f"Error-analysis data does not match the frozen experiment: {name}")
    elif args.protocol:
        raise FileNotFoundError(protocol_path)
    audit_path = args.data_audit or args.report_dir / "data_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else None
    if args.data_audit and audit is None:
        raise FileNotFoundError(audit_path)
    if audit and audit["manifest_sha256"] != digest(args.data_dir / "manifest.json"):
        raise ValueError("Metadata audit refers to another prepared-data manifest")
    for relative_path, expected in summary.get("provenance", {}).get("source_sha256", {}).items():
        path = PROJECT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if digest(path) != expected:
            raise ValueError(f"Frozen source differs before generating evidence: {relative_path}")
    analysis = error_analysis(summary, per_query, scores, args.data_dir, int(config["seed"]))
    image_paths = figures(summary, args.figures_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "error_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    args.output.write_text(results_document(summary, analysis, image_paths, args.output, digest(summary_path), protocol, audit))
    print(f"Generated {args.output}; figures: {args.figures_dir}; post-hoc examples: {args.output.parent / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
