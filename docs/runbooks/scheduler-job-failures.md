# Scheduler job failures

## Symptoms
"job failed", transient network errors, "IOError writing checkpoint", jobs retrying repeatedly or stuck.

## Likely causes
A dependent resource failing (disk, database or network), bad input data, timeouts, or overlapping runs of the same job.

## Diagnosis
Read the job's own logs, check the resources it depends on, look at retry counts, and check whether schedules overlap.

## Fix
Fix the failing dependency first, then rerun the job. Jobs should be idempotent so a rerun is safe. Clear a stuck lock if one exists.

## Prevention
Design idempotent jobs, retry with backoff, and send repeated failures to a dead-letter queue.
