-- queries.sql
-- Run against SQLite table `transactions` (loaded from financial_data.csv)
-- PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
--                 nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

-- 1. Fraud rate by transaction type
--    (Known fact about this dataset: fraud ONLY occurs in TRANSFER and CASH_OUT)
SELECT
    type,
    COUNT(*)                                   AS total_txns,
    SUM(isFraud)                                AS fraud_txns,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct
FROM transactions
GROUP BY type
ORDER BY fraud_rate_pct DESC;

-- 2. Balance-mismatch red flag: sender's account isn't debited correctly
--    (oldbalanceOrg - amount should equal newbalanceOrig; big gaps are a red flag)
SELECT
    nameOrig,
    type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    ROUND(oldbalanceOrg - amount - newbalanceOrig, 2) AS orig_balance_error,
    isFraud
FROM transactions
WHERE ABS(oldbalanceOrg - amount - newbalanceOrig) > 1
ORDER BY orig_balance_error DESC
LIMIT 50;

-- 3. Destination account emptied immediately after receiving funds
--    (classic "mule account" pattern: money in, then straight back out)
SELECT
    nameDest,
    COUNT(*)         AS times_used_as_dest,
    SUM(isFraud)      AS fraud_count
FROM transactions
GROUP BY nameDest
HAVING fraud_count > 0
ORDER BY fraud_count DESC
LIMIT 20;

-- 4. Average transaction amount: fraud vs non-fraud
SELECT
    isFraud,
    COUNT(*)                          AS n_txns,
    ROUND(AVG(amount), 2)              AS avg_amount,
    ROUND(MAX(amount), 2)              AS max_amount
FROM transactions
GROUP BY isFraud;
