# Slow database queries

## Symptoms
Messages like "slow query took 1200ms", rising latency, high database CPU, and lock waits. Slow queries often appear minutes before a connection pool is exhausted.

## Likely causes
A missing index leading to full table scans, N+1 query patterns, lock contention, or outdated table statistics.

## Diagnosis
Run EXPLAIN ANALYZE on the query, enable the slow query log, check query statistics for the most expensive statements, and look for blocked sessions.

## Fix
Add the missing index, rewrite or batch the query, refresh statistics, and shorten long transactions.

## Prevention
Review queries before release, alert on slow query counts, and load test new endpoints.
