"""Shared train-only lexical encoder and typed-attribute representation."""

from functools import lru_cache
import re
import time

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from .attributes import FEATURE_NAMES as ATTRIBUTE_NAMES, compare_attributes
from .text import document, tokens

LEXICAL_NAMES = (
    "title_tfidf", "body_tfidf", "title_char_tfidf", "title_bm25", "body_bm25",
    "query_tokens", "query_chars", "title_tokens", "body_tokens", "title_chars",
    "title_coverage", "body_coverage", "title_jaccard", "body_jaccard",
    "title_idf_coverage", "body_idf_coverage", "title_phrase", "body_phrase",
    "brand_mentioned", "brand_token_coverage", "query_numeric_count",
    "title_numeric_coverage", "body_numeric_coverage", "query_has_negation",
    "title_prefix_coverage", "title_suffix_coverage", "body_empty", "brand_empty",
)
ALL_NAMES = LEXICAL_NAMES + ATTRIBUTE_NAMES


@lru_cache(maxsize=20_000)
def _parsed(query, title, body):
    # Training needs thirty numbers, not the much larger nested UI evidence.
    return tuple(compare_attributes(query, title, body)["feature_vector"])


class FeatureEncoder:
    def __init__(self, word_features=60000, character_features=40000):
        self.word = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False,
            norm=None, dtype=np.float32, max_features=word_features, min_df=2)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
            lowercase=True, strip_accents="unicode", dtype=np.float32,
            max_features=character_features, min_df=3)

    def fit(self, products):
        docs = [document(*v) for v in products.select("title", "brand", "description", "attributes").iter_rows()]
        titles = [v[0] for v in docs]
        bodies = [" ".join(v) for v in docs]
        self.word.fit(bodies)
        self.char.fit(titles)
        self.n_fit_documents = len(docs)
        df = (len(docs) + 1) / np.exp(self.word.idf_ - 1) - 1
        self.bm25_idf = np.log1p((len(docs) - df + .5) / (df + .5)).astype(np.float32)
        self.avg_title = max(np.mean([len(tokens(t)) for t in titles]), 1)
        self.avg_body = max(np.mean([len(tokens(t)) for t in bodies]), 1)
        self.idf_lookup = dict(zip(self.word.get_feature_names_out(), map(float, self.word.idf_)))
        return self

    def _require_fitted(self):
        check_is_fitted(self.word)
        check_is_fitted(self.char)
        if not all(hasattr(self, name) for name in ("bm25_idf", "avg_title", "avg_body", "idf_lookup")):
            raise NotFittedError("Fit FeatureEncoder on reference products before transforming or retrieving")

    def _bank(self, products):
        docs = [document(*v) for v in products.select("title", "brand", "description", "attributes").iter_rows()]
        title = [v[0] for v in docs]
        body = [" ".join(v) for v in docs]
        title_raw, body_raw = self.word.transform(title), self.word.transform(body)
        lens_t = np.array([len(tokens(x)) for x in title], dtype=np.float32)
        lens_b = np.array([len(tokens(x)) for x in body], dtype=np.float32)

        def bm25(raw, lengths, average):
            out = raw.copy().tocsr()
            counts = out.data / self.word.idf_[out.indices]
            length_norm = np.repeat(1.2 * (.25 + .75 * lengths / average), np.diff(out.indptr))
            out.data = (self.bm25_idf[out.indices] * counts * 2.2 / (counts + length_norm)).astype(np.float32)
            return out

        return {"docs": docs, "title": normalize(title_raw), "body": normalize(body_raw),
            "char": self.char.transform(title), "title_bm25": bm25(title_raw, lens_t, self.avg_title),
            "body_bm25": bm25(body_raw, lens_b, self.avg_body)}

    def transform(self, pairs, *, progress=False):
        self._require_fitted()
        started = time.perf_counter()
        # IDs only index identical texts; they are never numerical model inputs.
        products = pairs.select("product_id", "title", "brand", "description", "attributes").unique(maintain_order=True)
        queries = pairs.select("query_id", "query").unique(maintain_order=True)
        if products["product_id"].null_count() or products["product_id"].n_unique() != products.height:
            raise ValueError("Each non-null product_id must identify one consistent product text")
        if queries["query_id"].null_count() or queries["query_id"].n_unique() != queries.height:
            raise ValueError("Each non-null query_id must identify one consistent query text")
        if any(not isinstance(q, str) for q in queries["query"]):
            raise ValueError("Query text must be a string")
        if not pairs.height:
            return np.zeros((0, len(ALL_NAMES)), dtype=np.float32)
        pmap = {v: i for i, v in enumerate(products["product_id"])}
        qmap = {v: i for i, v in enumerate(queries["query_id"])}
        pi = np.array([pmap[v] for v in pairs["product_id"]], dtype=np.int32)
        qi = np.array([qmap[v] for v in pairs["query_id"]], dtype=np.int32)
        bank = self._bank(products)
        qtexts = queries["query"].to_list()
        qraw = self.word.transform(qtexts)
        qw = normalize(qraw)
        qb = qraw.copy()
        qb.data.fill(1)
        qc = self.char.transform(qtexts)
        x = np.zeros((pairs.height, len(ALL_NAMES)), dtype=np.float32)
        for a in range(0, pairs.height, 10000):
            b = min(a + 10000, pairs.height)
            ps, qs = pi[a:b], qi[a:b]
            for j, (matrix, query_matrix) in enumerate(((bank["title"], qw), (bank["body"], qw), (bank["char"], qc), (bank["title_bm25"], qb), (bank["body_bm25"], qb))):
                x[a:b, j] = np.asarray(matrix[ps].multiply(query_matrix[qs]).sum(axis=1)).ravel()
        qinfo = []
        for q in qtexts:
            tt = tokens(q); ss = set(tt)
            nums = set(re.findall(r"\d+(?:[.,]\d+)?", q.lower()))
            qinfo.append((q, tt, ss, nums, sum(self.idf_lookup.get(t, 1.0) for t in ss)))
        pinfo = []
        for title, brand, rest in bank["docs"]:
            body = " ".join((title, brand, rest))
            tt, bt = tokens(title), tokens(body)
            pinfo.append((title, brand, body, set(tt), set(bt), len(tt), len(bt),
                set(re.findall(r"\d+(?:[.,]\d+)?", title.lower())), set(re.findall(r"\d+(?:[.,]\d+)?", body.lower()))))
        for i, (pn, qn) in enumerate(zip(pi, qi)):
            q, qt, qs, qnums, weight = qinfo[qn]
            title, brand, body, ts, bs, tl, bl, tn, bn = pinfo[pn]
            it, ib = qs & ts, qs & bs
            cov = lambda a, b: len(a & b) / max(len(a), 1)
            qlower = q.lower()
            bt = set(tokens(brand))
            lexical = [len(qt), len(q), tl, bl, len(title), cov(qs, ts), cov(qs, bs),
                len(it) / max(len(qs | ts), 1), len(ib) / max(len(qs | bs), 1),
                sum(self.idf_lookup.get(t, 1.0) for t in it) / max(weight, 1),
                sum(self.idf_lookup.get(t, 1.0) for t in ib) / max(weight, 1),
                float(bool(q.strip()) and qlower in title.lower()), float(bool(q.strip()) and qlower in body.lower()),
                float(bool(brand) and brand.lower() in qlower), len(qs & bt) / max(len(bt), 1), len(qnums),
                cov(qnums, tn), cov(qnums, bn), float(bool(qs & {"no", "not", "without", "except"})),
                sum(any(v.startswith(t) for v in ts) for t in qs) / max(len(qs), 1),
                sum(any(v.endswith(t) for v in ts) for t in qs) / max(len(qs), 1), float(not body.strip()), float(not brand.strip())]
            x[i, 5:len(LEXICAL_NAMES)] = lexical
            x[i, len(LEXICAL_NAMES):] = _parsed(q, title, body)
            if progress and i and i % 100000 == 0:
                print(f"Features {i:,}/{pairs.height:,}; {time.perf_counter()-started:.1f}s", flush=True)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite feature")
        return x

    def retrieval_index(self, products):
        self._require_fitted()
        docs = [" ".join(document(*v)) for v in products.select("title", "brand", "description", "attributes").iter_rows()]
        raw = self.word.transform(docs)
        counts = raw.data / self.word.idf_[raw.indices]
        lengths = np.array([len(tokens(x)) for x in docs], np.float32)
        norm = np.repeat(1.2 * (.25 + .75 * lengths / self.avg_body), np.diff(raw.indptr))
        raw.data = (self.bm25_idf[raw.indices] * counts * 2.2 / (counts + norm)).astype(np.float32)
        return raw.tocsr()

    def retrieve(self, query, index, limit=150):
        self._require_fitted()
        if not isinstance(limit, (int, np.integer)) or limit < 1:
            raise ValueError("limit must be a positive integer")
        q = self.word.transform([query])
        q.data.fill(1)
        scores = np.asarray((index @ q.T).toarray()).ravel()
        nonzero = np.flatnonzero(scores > 0)
        if len(nonzero) > limit:
            nonzero = nonzero[np.argpartition(-scores[nonzero], limit - 1)[:limit]]
        return nonzero[np.lexsort((nonzero, -scores[nonzero]))], scores
