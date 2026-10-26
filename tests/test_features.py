"""Encoder invariants on a coherent, fully synthetic catalog."""

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from catalog_search.attributes import compare_attributes
from catalog_search.features import ALL_NAMES, ATTRIBUTE_NAMES, LEXICAL_NAMES, FeatureEncoder
from catalog_search.text import document


@pytest.fixture
def products():
    return pl.DataFrame({
        "product_id": [f"p{i}" for i in range(6)],
        "title": ["Samsung Galaxy S21 128 GB phone", "Samsung Galaxy S21 256 GB phone",
                  "Anker USB charger 65 W", "Anker USB charger 100 W",
                  "Phone case for iPhone 11 Pro", "Phone case for iPhone 11"],
        "brand": ["Samsung", "Samsung", "Anker", "Anker", "", ""],
        "description": ["Black Android smartphone", "Blue Android smartphone",
                        "USB power adapter for phones", "USB power adapter for phones",
                        "Slim phone protective cover", "Slim phone protective cover"],
        "attributes": ["Storage: 128 GB", "Storage: 256 GB", "Output: 65 W",
                       "Output: 100 W", "", ""],
    })


@pytest.fixture
def pairs(products):
    rows = pl.DataFrame({
        "query_id": ["q0", "q0", "q0", "q1", "q1", "q2", "q2"],
        "query": ["Samsung 128 GB phone"] * 3 + ["USB charger"] * 2 + ["iPhone 11 case"] * 2,
        "product_id": ["p0", "p1", "p5", "p2", "p3", "p4", "p5"],
    })
    return rows.join(products, on="product_id", how="left", maintain_order="left")


@pytest.fixture
def encoder(products):
    return FeatureEncoder(word_features=1000, character_features=1000).fit(products)


def test_dimensions_finiteness_and_attribute_parity(encoder, pairs):
    x = encoder.transform(pairs)
    assert len(LEXICAL_NAMES) == 28
    assert len(ATTRIBUTE_NAMES) == 30
    assert ALL_NAMES == LEXICAL_NAMES + ATTRIBUTE_NAMES
    assert x.shape == (pairs.height, 58)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    for i, row in enumerate(pairs.iter_rows(named=True)):
        title, brand, rest = document(row["title"], row["brand"], row["description"], row["attributes"])
        body = " ".join((title, brand, rest))
        expected = compare_attributes(row["query"], title, body)["feature_vector"]
        np.testing.assert_allclose(x[i, 28:], expected)


def test_row_order_and_batch_partition_do_not_change_features(encoder, pairs):
    full = encoder.transform(pairs)
    order = [6, 0, 4, 1, 3, 5, 2]
    shuffled = encoder.transform(pairs[order])
    np.testing.assert_allclose(shuffled, full[order], rtol=1e-6, atol=1e-7)
    batched = np.concatenate([encoder.transform(pairs[:2]), encoder.transform(pairs[2:5]), encoder.transform(pairs[5:])])
    np.testing.assert_allclose(batched, full, rtol=1e-6, atol=1e-7)
    duplicated = encoder.transform(pl.concat([pairs, pairs[:1]]))
    np.testing.assert_allclose(duplicated[-1], full[0])


def test_unseen_tokens_never_refit_vocabulary_or_idf(encoder, pairs):
    word_vocab, char_vocab = dict(encoder.word.vocabulary_), dict(encoder.char.vocabulary_)
    word_idf, char_idf = encoder.word.idf_.copy(), encoder.char.idf_.copy()
    new_pairs = pairs[:1].with_columns(pl.lit("new-query").alias("query_id"), pl.lit("zzzxylophonium qvwxz").alias("query"))
    assert np.isfinite(encoder.transform(new_pairs)).all()
    assert encoder.word.vocabulary_ == word_vocab
    assert encoder.char.vocabulary_ == char_vocab
    np.testing.assert_array_equal(encoder.word.idf_, word_idf)
    np.testing.assert_array_equal(encoder.char.idf_, char_idf)


def test_bm25_retrieval_has_actual_positive_hits_and_stable_ties(encoder, products):
    index = encoder.retrieval_index(products)
    hits, scores = encoder.retrieve("USB charger", index, limit=6)
    assert hits.tolist() == [2, 3]
    assert scores[2] > 0 and scores[3] > 0
    assert not np.any(scores[[0, 1, 4, 5]] > 0)
    single, _ = encoder.retrieve("USB charger", index, limit=1)
    assert single.tolist() == [2]
    absent, zero_scores = encoder.retrieve("zzzxylophonium", index)
    assert absent.size == 0 and np.all(zero_scores == 0)
    with pytest.raises(ValueError, match="positive integer"):
        encoder.retrieve("USB", index, limit=0)


def test_unfitted_word_char_and_encoder_operations_raise(products, pairs):
    model = FeatureEncoder()
    with pytest.raises(NotFittedError):
        model.word.transform(["phone"])
    with pytest.raises(NotFittedError):
        model.char.transform(["phone"])
    with pytest.raises(NotFittedError):
        model.transform(pairs)
    with pytest.raises(NotFittedError):
        model.retrieval_index(products)


def test_empty_batch_retains_shape_and_id_text_collisions_raise(encoder, pairs):
    assert encoder.transform(pairs.head(0)).shape == (0, 58)
    bad_product = pl.concat([pairs[:1], pairs[:1].with_columns(pl.lit("different text").alias("title"))])
    with pytest.raises(ValueError, match="product_id"):
        encoder.transform(bad_product)
    bad_query = pl.concat([pairs[:1], pairs[:1].with_columns(pl.lit("different query").alias("query"))])
    with pytest.raises(ValueError, match="query_id"):
        encoder.transform(bad_query)


def test_empty_query_has_no_overlap_or_retrieval_signal(encoder, pairs, products):
    empty = pairs[:1].with_columns(pl.lit("").alias("query"))
    x = encoder.transform(empty)
    np.testing.assert_array_equal(x[0, :5], np.zeros(5))
    assert x[0, LEXICAL_NAMES.index("query_tokens")] == 0
    hits, scores = encoder.retrieve("", encoder.retrieval_index(products))
    assert hits.size == 0 and np.all(scores == 0)
