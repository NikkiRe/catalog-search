SELECT 'duplicate_scored_pair' AS contract, COUNT(*) AS violations FROM (
    SELECT query_id, product_id FROM judged GROUP BY query_id, product_id HAVING COUNT(*) > 1
)
UNION ALL
SELECT 'invalid_label', COUNT(*) FROM judged WHERE label NOT IN ('E', 'S', 'C', 'I') OR label IS NULL
UNION ALL
SELECT 'inconsistent_query_family', COUNT(*) FROM (
    SELECT query_id FROM judged GROUP BY query_id HAVING COUNT(DISTINCT family) <> 1 OR COUNT(family) <> COUNT(*)
)
UNION ALL
SELECT 'duplicate_tie_within_query', COUNT(*) FROM (
    SELECT query_id, tie_key FROM judged GROUP BY query_id, tie_key HAVING COUNT(*) > 1
);
