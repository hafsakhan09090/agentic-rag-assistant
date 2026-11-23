# Cache eviction failures

## Symptoms
"eviction failed for key", a falling cache hit ratio, high memory use in the cache, and extra load on the database.

## Likely causes
Entries stored without a TTL, very large keys, a memory limit that is too small, or an eviction policy that does not match the workload.

## Diagnosis
Check cache memory statistics and the number of evicted keys, inspect the TTL distribution, and find the largest keys.

## Fix
Set TTLs on entries, choose an appropriate eviction policy such as least recently used, increase memory, and remove oversized keys.

## Prevention
Monitor the hit ratio and memory, and review which data is cached.
