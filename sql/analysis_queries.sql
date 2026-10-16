-- Each view answers one business question and exposes the same simple output
-- contract: dimension, metric, transaction_count.

DROP VIEW IF EXISTS age_band_average_price;
CREATE VIEW age_band_average_price AS
SELECT
    CASE
        WHEN t.age IS NULL THEN 'Unknown'
        WHEN t.age < 5 THEN '0-4 years'
        WHEN t.age < 10 THEN '5-9 years'
        WHEN t.age < 20 THEN '10-19 years'
        ELSE '20+ years'
    END AS dimension,
    ROUND(AVG(t.target_value), 2) AS metric,
    COUNT(*) AS transaction_count
FROM equipment_transactions AS t
GROUP BY dimension
ORDER BY CASE dimension
    WHEN '0-4 years' THEN 1
    WHEN '5-9 years' THEN 2
    WHEN '10-19 years' THEN 3
    WHEN '20+ years' THEN 4
    ELSE 5
END;

DROP VIEW IF EXISTS category_median_price;
CREATE VIEW category_median_price AS
WITH ranked AS (
    SELECT
        COALESCE(
            NULLIF(a.inventory_group_description, ''),
            a.inventory_group_category,
            'Unknown'
        ) AS category,
        t.target_value,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(
                NULLIF(a.inventory_group_description, ''),
                a.inventory_group_category,
                'Unknown'
            )
            ORDER BY t.target_value
        ) AS row_number_in_category,
        COUNT(*) OVER (
            PARTITION BY COALESCE(
                NULLIF(a.inventory_group_description, ''),
                a.inventory_group_category,
                'Unknown'
            )
        ) AS category_count
    FROM equipment_transactions AS t
    JOIN equipment_attributes AS a USING (asset_id)
)
SELECT
    category AS dimension,
    ROUND(AVG(target_value), 2) AS metric,
    MAX(category_count) AS transaction_count
FROM ranked
WHERE row_number_in_category IN (
    CAST((category_count + 1) / 2 AS INTEGER),
    CAST((category_count + 2) / 2 AS INTEGER)
)
GROUP BY category
ORDER BY metric DESC;

DROP VIEW IF EXISTS region_average_price;
CREATE VIEW region_average_price AS
SELECT
    COALESCE(t.region_code, 'Unknown') AS dimension,
    ROUND(AVG(t.target_value), 2) AS metric,
    COUNT(*) AS transaction_count
FROM equipment_transactions AS t
GROUP BY COALESCE(t.region_code, 'Unknown')
ORDER BY metric DESC;

DROP VIEW IF EXISTS utilization_average_price;
CREATE VIEW utilization_average_price AS
SELECT
    COALESCE(t.utilization_tier, 'Unknown') AS dimension,
    ROUND(AVG(t.target_value), 2) AS metric,
    COUNT(*) AS transaction_count
FROM equipment_transactions AS t
GROUP BY COALESCE(t.utilization_tier, 'Unknown')
ORDER BY metric DESC;

DROP VIEW IF EXISTS operational_hours_average_price;
CREATE VIEW operational_hours_average_price AS
SELECT
    CASE
        WHEN t.operational_hours_meter IS NULL THEN 'Unknown'
        WHEN t.operational_hours_meter = 0 THEN '0 hours'
        WHEN t.operational_hours_meter < 2000 THEN '1-1,999 hours'
        WHEN t.operational_hours_meter < 5000 THEN '2,000-4,999 hours'
        WHEN t.operational_hours_meter < 10000 THEN '5,000-9,999 hours'
        ELSE '10,000+ hours'
    END AS dimension,
    ROUND(AVG(t.target_value), 2) AS metric,
    COUNT(*) AS transaction_count
FROM equipment_transactions AS t
GROUP BY dimension
ORDER BY CASE dimension
    WHEN '0 hours' THEN 1
    WHEN '1-1,999 hours' THEN 2
    WHEN '2,000-4,999 hours' THEN 3
    WHEN '5,000-9,999 hours' THEN 4
    WHEN '10,000+ hours' THEN 5
    ELSE 6
END;

DROP VIEW IF EXISTS category_transaction_volume;
CREATE VIEW category_transaction_volume AS
SELECT
    COALESCE(
        NULLIF(a.inventory_group_description, ''),
        a.inventory_group_category,
        'Unknown'
    ) AS dimension,
    CAST(COUNT(*) AS REAL) AS metric,
    COUNT(*) AS transaction_count
FROM equipment_transactions AS t
JOIN equipment_attributes AS a USING (asset_id)
GROUP BY COALESCE(
    NULLIF(a.inventory_group_description, ''),
    a.inventory_group_category,
    'Unknown'
)
ORDER BY transaction_count DESC;

-- Human-readable examples. The Python runner reads the views above and adds
-- an analysis name before combining them into one reproducible CSV report.
SELECT * FROM age_band_average_price;
SELECT * FROM category_median_price;
SELECT * FROM region_average_price;
SELECT * FROM utilization_average_price;
SELECT * FROM operational_hours_average_price;
SELECT * FROM category_transaction_volume;
