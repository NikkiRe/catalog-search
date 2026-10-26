-- Independent ESCI metric implementation. Gains are utilities, not 2^label-1.
-- tie_key is the 16-digit hex encoding of the unsigned 64-bit hash.
WITH long_scores AS (
    SELECT query_id, product_id, family, label, tie_key, 'bm25' AS method, bm25 AS score FROM judged
    UNION ALL SELECT query_id, product_id, family, label, tie_key, 'tfidf', tfidf FROM judged
    UNION ALL SELECT query_id, product_id, family, label, tie_key, 'baseline', baseline FROM judged
    UNION ALL SELECT query_id, product_id, family, label, tie_key, 'rule', rule FROM judged
    UNION ALL SELECT query_id, product_id, family, label, tie_key, 'no_product_evidence', no_product_evidence FROM judged
    UNION ALL SELECT query_id, product_id, family, label, tie_key, 'attribute', attribute FROM judged
), graded AS (
    SELECT *, CASE label WHEN 'E' THEN 1.0 WHEN 'S' THEN 0.1
        WHEN 'C' THEN 0.01 WHEN 'I' THEN 0.0 END AS gain
    FROM long_scores
), ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY query_id, method ORDER BY score DESC, tie_key) AS position,
        ROW_NUMBER() OVER (PARTITION BY query_id, method ORDER BY gain DESC, tie_key) AS ideal_position
    FROM graded
), discounted AS (
    SELECT query_id, family, method,
        SUM(CASE WHEN position <= 10 THEN gain / log2(position + 1.0) ELSE 0.0 END) AS dcg10,
        SUM(CASE WHEN ideal_position <= 10 THEN gain / log2(ideal_position + 1.0) ELSE 0.0 END) AS idcg10,
        COUNT(*) AS candidates
    FROM ranked GROUP BY query_id, family, method
)
SELECT query_id, family, method, candidates, dcg10, idcg10,
    CASE WHEN idcg10 > 0 THEN dcg10 / idcg10 ELSE 0.0 END AS ndcg10
FROM discounted ORDER BY query_id, method;
