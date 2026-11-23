# Out of memory

## Symptoms
"out of memory", the OOM killer ending a process, "cannot allocate memory", containers restarting with exit code 137, heavy swapping, or "eviction pressure high" warnings.

## Likely causes
A memory leak, an unbounded cache or queue, unusually large payloads, or memory limits set too low for the workload.

## Diagnosis
Search the kernel log with dmesg for OOM messages. Check container memory limits and the memory trend over time. Take a heap profile if the language supports it. A steady climb suggests a leak; a sudden jump suggests a large request.

## Fix
As a stopgap, restart with a higher limit. Then fix the leak, put bounds on caches and queues, and limit payload sizes.

## Prevention
Alert on memory growth, set realistic limits, and include memory in load tests.
