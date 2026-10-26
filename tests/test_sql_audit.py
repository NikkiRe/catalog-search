"""Hand calculations and damaged data for the independent SQL implementation."""

import importlib.util
import math
from pathlib import Path
import sqlite3

import polars as pl
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("sql_audit", ROOT / "scripts/sql_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _row(query, product, label, score, tie):
    return {"query_id": query, "product_id": product, "family": query, "label": label,
            "tie_order": tie, **{mode: score for mode in audit.MODES}}


def _calculate(rows):
    frame = pl.DataFrame(rows, schema_overrides={"tie_order": pl.UInt64})
    with sqlite3.connect(":memory:") as connection:
        connection.create_function("log2", 1, math.log2)
        audit.load_scores(connection, frame)
        audit.check_contracts(connection, ROOT / "sql")
        return connection.execute((ROOT / "sql/ndcg.sql").read_text()).fetchall()


def test_sql_linear_gains_against_hand_calculation():
    rows = [_row("q", str(i), label, score, i) for i, (label, score) in
            enumerate((("E", 0), ("S", 3), ("C", 2), ("I", 1)))]
    expected = (.1 + .01 / math.log2(3) + 1 / math.log2(5)) / (1 + .1 / math.log2(3) + .01 / 2)
    values = _calculate(rows)
    assert len(values) == 6
    assert all(row[-1] == pytest.approx(expected, abs=1e-14) for row in values)


def test_sql_cutoff_zero_idcg_and_query_weighting():
    rows = [_row("perfect", "e", "E", 1, 0), _row("zero", "i", "I", 1, 0)]
    rows += [_row("long", str(i), "E" if i == 15 else "I", -i, i) for i in range(16)]
    result = _calculate(rows)
    baseline = {row[0]: row[-1] for row in result if row[2] == "baseline"}
    assert baseline == {"perfect": 1, "zero": 0, "long": 0}
    assert sum(baseline.values()) / len(baseline) == pytest.approx(1 / 3)


def test_sql_unsigned_64bit_ties_preserve_order_and_input_permutation():
    rows = [_row("q", "low", "E", 0, 2**63 - 1), _row("q", "high", "I", 0, 2**64 - 1)]
    assert _calculate(rows) == _calculate(rows[::-1])
    assert all(row[-1] == 1 for row in _calculate(rows))


@pytest.mark.parametrize("mutation", ["duplicate", "invalid_label", "inconsistent_family", "duplicate_tie"])
def test_sql_contracts_reject_corrupt_candidate_lists(mutation):
    rows = [_row("q", "a", "E", 2, 0), _row("q", "b", "I", 1, 1)]
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "invalid_label":
        rows[0]["label"] = "unknown"
    elif mutation == "inconsistent_family":
        rows[1]["family"] = "other"
    else:
        rows[1]["tie_order"] = 0
    with pytest.raises(ValueError, match="contracts"):
        _calculate(rows)


def test_sql_joins_detect_query_family_leak_and_missing_product(tmp_path):
    pl.DataFrame({"query_id": ["q", "dev"], "family": ["q", "q"],
                  "partition": ["test", "dev"]}).write_parquet(tmp_path / "queries.parquet")
    pl.DataFrame({"product_locale": ["us"], "product_id": ["other"]}).write_parquet(tmp_path / "products.parquet")
    with sqlite3.connect(":memory:") as connection:
        audit.load_scores(connection, pl.DataFrame([_row("q", "missing", "E", 1, 0)]))
        with pytest.raises(ValueError, match="family_crosses_partitions.*missing_product_metadata"):
            audit.check_contracts(connection, ROOT / "sql", tmp_path)
