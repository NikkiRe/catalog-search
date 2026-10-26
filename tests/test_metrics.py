"""Independent, hand-calculated metric checks; no model-performance evaluation."""

import math

import numpy as np
import polars as pl
import pytest

from catalog_search.metrics import ESCI_GAINS, evaluate, ndcg, paired_bootstrap
from catalog_search.text import stable_hash


def test_esci_gain_contract_and_hand_calculated_dcg():
    assert ESCI_GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
    # Score order S, C, I, E. Gains are already utilities, not ordinal labels.
    observed = .1 + .01 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + .1 / math.log2(3) + .01 / 2
    assert ndcg([1, .1, .01, 0], [0, 3, 2, 1], k=4) == pytest.approx(observed / ideal)
    assert ndcg([1, .1, .01, 0], [4, 3, 2, 1], k=4) == pytest.approx(1)


def test_cutoff_empty_and_zero_idcg_conventions():
    assert ndcg([1, 0], [0, 1], k=1) == 0
    assert ndcg([1, 0], [0, 1], k=2) == pytest.approx(1 / math.log2(3))
    assert ndcg([1, 0], [0, 1], k=100) == ndcg([1, 0], [0, 1], k=2)
    assert ndcg([0, 0], [99, -50]) == 0
    assert ndcg([], []) == 0


def test_label_blind_hash_ties_are_invariant_to_input_permutation():
    products = ["product-a", "product-b", "product-c", "product-d"]
    gains = np.array([1., .1, .01, 0.])
    ties = np.array([stable_hash(f"q|{product}") for product in products], dtype=np.uint64)
    order = sorted(range(4), key=lambda index: int(ties[index]))
    expected_dcg = sum(gains[index] / math.log2(rank + 2) for rank, index in enumerate(order))
    ideal = 1 + .1 / math.log2(3) + .01 / 2
    original = ndcg(gains, np.zeros(4), k=4, tie_order=ties)
    assert original == pytest.approx(expected_dcg / ideal)
    permutation = [3, 1, 0, 2]
    assert ndcg(gains[permutation], np.zeros(4), k=4, tie_order=ties[permutation]) == original


@pytest.mark.parametrize("gains,scores", [([1], [float("nan")]), ([1], [float("inf")]), ([1, 0], [1])])
def test_nonfinite_scores_and_misaligned_inputs_fail(gains, scores):
    with pytest.raises(ValueError):
        ndcg(gains, scores)


@pytest.mark.parametrize("gains", [[float("nan"), 1], [float("inf"), 1], [-.1, 1]])
def test_invalid_gains_are_rejected_instead_of_silently_becoming_zero(gains):
    with pytest.raises(ValueError):
        ndcg(gains, [1, 0])


@pytest.mark.parametrize("cutoff", [0, -1])
def test_nonpositive_cutoff_is_not_python_negative_slicing(cutoff):
    with pytest.raises(ValueError):
        ndcg([1, .1, 0], [0, 1, 2], k=cutoff)


def test_evaluate_weights_queries_equally_and_keeps_zero_idcg_queries():
    rows = [{"query_id": "short", "family": "short", "label": "E", "tie_order": 0,
             "baseline": 1., "attribute": 1.}]
    for index in range(20):
        rows.append({"query_id": "long", "family": "long", "label": "E" if index == 19 else "I",
                     "tie_order": index, "baseline": float(index), "attribute": float(-index)})
    rows += [{"query_id": "no-relevant", "family": "no-relevant", "label": "I", "tie_order": 0,
              "baseline": 1., "attribute": 1.}]
    means, per_query = evaluate(pl.DataFrame(rows), ["baseline", "attribute"])
    assert per_query.height == 3
    assert means["baseline"]["ndcg10"] == pytest.approx(2 / 3)
    assert means["attribute"]["ndcg10"] == pytest.approx(1 / 3)
    assert means["attribute"]["exact_recall10"] == pytest.approx(.5)
    assert per_query.filter(pl.col("query_id") == "no-relevant")["zero_idcg"].item()
    assert per_query.filter(pl.col("query_id") == "no-relevant")["attribute_exact_recall10"].item() is None
    assert per_query.filter(pl.col("query_id") == "long")["attribute_mrr"].item() == pytest.approx(1 / 20)


def test_bootstrap_point_estimate_is_query_weighted_not_family_weighted():
    frame = pl.DataFrame({"family": ["a", "a", "a", "b"],
                          "baseline_ndcg10": [0., 0., 0., .1],
                          "attribute_ndcg10": [.2, .3, .4, 0.]})
    result = paired_bootstrap(frame, repeats=1000, seed=7)
    assert result["difference"] == pytest.approx(.2)  # (.2+.3+.4-.1) / 4
    assert result["difference"] != pytest.approx(.1)  # mean of two family means
    assert result["ci_low"] == pytest.approx(-.1)
    assert result["ci_high"] == pytest.approx(.3)
    assert result["queries"] == 4 and result["families"] == 2


def test_bootstrap_single_family_and_crossing_zero_difference():
    frame = pl.DataFrame({"family": ["one", "one"], "baseline_ndcg10": [.25, .75],
                          "attribute_ndcg10": [.75, .25]})
    result = paired_bootstrap(frame, repeats=20, seed=9)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
    # This tests computational semantics; one family cannot justify population precision.


def test_bootstrap_seed_reproduces_ci_after_row_reordering():
    rows = []
    for family in range(13):
        for _ in range(1 + family % 4):
            rows.append({"family": f"family-{family:02d}", "baseline_ndcg10": .25,
                         "attribute_ndcg10": .25 + family / 64})
    frame = pl.DataFrame(rows)
    first = paired_bootstrap(frame, repeats=137, seed=20261026)
    assert paired_bootstrap(frame, repeats=137, seed=20261026) == first
    assert paired_bootstrap(frame.reverse(), repeats=137, seed=20261026) == first


def test_bootstrap_empty_groups_and_invalid_repeat_count_fail():
    empty = pl.DataFrame(schema={"family": pl.String, "baseline_ndcg10": pl.Float64,
                                "attribute_ndcg10": pl.Float64})
    with pytest.raises(ValueError, match="famil"):
        paired_bootstrap(empty)
    frame = pl.DataFrame({"family": ["a"], "baseline_ndcg10": [.5], "attribute_ndcg10": [.6]})
    with pytest.raises(ValueError):
        paired_bootstrap(frame, repeats=0)


def test_training_callback_matches_hand_calculation_with_ties_and_zero_idcg():
    from catalog_search.experiment import _GroupedNDCG
    labels = np.array([3, 2, 0, 0, 0, 3, 1], dtype=np.int32)
    groups = np.array([3, 2, 2], dtype=np.int32)
    ties = np.array([0, 1, 2, 0, 1, 9, 1], dtype=np.uint64)
    scores = np.array([0., 1., 2., 1., 0., 0., 0.])
    callback = _GroupedNDCG(labels, groups, ties, k=2)
    # q1 orders I,S,E but cutoff2 excludes E; q2 has no relevance;
    # q3 ties resolve C before E using the supplied label-blind order.
    first = (.1 / math.log2(3)) / (1 + .1 / math.log2(3))
    third = (.01 + 1 / math.log2(3)) / (1 + .01 / math.log2(3))
    name, value, maximize = callback(labels.astype(np.float32), scores, None, groups)
    assert name == "protocol_ndcg10" and maximize
    assert value == pytest.approx((first + third) / 3)
    with pytest.raises(ValueError, match="order"):
        callback(labels[::-1], scores, None, groups)


def test_experiment_group_guard_prevents_feature_label_group_misalignment():
    from catalog_search.experiment import _groups
    assert _groups(pl.DataFrame({"query_id": ["a", "a", "b"]})).tolist() == [2, 1]
    with pytest.raises(ValueError, match="contiguous"):
        _groups(pl.DataFrame({"query_id": ["a", "b", "a"]}))


def test_query_only_ablation_excludes_every_product_comparison_feature():
    from catalog_search.experiment import _variants
    from catalog_search.features import ALL_NAMES
    variants = _variants({"include_query_only_ablation": True})
    assert len(variants["baseline"]) == 28 and len(variants["attribute"]) == 58
    query_only = [ALL_NAMES[index] for index in variants["no_product_evidence"] if index >= 28]
    assert set(query_only) == {"attr_query_count", "attr_ambiguous_query_count",
                              "attr_storage_query_count", "attr_power_query_count", "attr_length_query_count",
                              "attr_model_query_count", "attr_brand_query_count"}
