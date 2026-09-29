-- 1. Churn rate by contract type
SELECT Contract,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn = 'Yes'), 1) AS churn_pct
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;

-- 2. Churn rate by tenure group
SELECT CASE WHEN tenure <= 12 THEN '0-12 months'
            WHEN tenure <= 36 THEN '13-36 months'
            ELSE '37+ months' END AS tenure_group,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn = 'Yes'), 1) AS churn_pct
FROM customers
GROUP BY tenure_group
ORDER BY churn_pct DESC;

-- 3. Annual revenue lost to churn, by contract
SELECT Contract,
       ROUND(SUM(MonthlyCharges) * 12, 0) AS annual_revenue_lost
FROM customers
WHERE Churn = 'Yes'
GROUP BY Contract
ORDER BY annual_revenue_lost DESC;