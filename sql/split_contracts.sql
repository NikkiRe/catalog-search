SELECT 'family_crosses_partitions' AS contract, COUNT(*) AS violations FROM (
    SELECT family FROM split_queries GROUP BY family HAVING COUNT(DISTINCT partition) > 1
)
UNION ALL
SELECT 'query_crosses_partitions', COUNT(*) FROM (
    SELECT query_id FROM split_queries GROUP BY query_id HAVING COUNT(*) <> 1
)
UNION ALL
SELECT 'scored_query_not_in_test', COUNT(*) FROM judged j
LEFT JOIN split_queries q ON j.query_id = q.query_id
WHERE q.query_id IS NULL OR q.partition <> 'test'
UNION ALL
SELECT 'scored_family_differs_from_split', COUNT(*) FROM judged j
JOIN split_queries q ON j.query_id = q.query_id WHERE j.family <> q.family
UNION ALL
SELECT 'missing_product_metadata', COUNT(*) FROM judged j
LEFT JOIN products p ON j.product_id = p.product_id AND p.product_locale = 'us'
WHERE p.product_id IS NULL;
