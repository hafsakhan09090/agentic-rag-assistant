# Upstream timeouts and 5xx errors

## Symptoms
"upstream timeout after 30000ms", 502 bad gateway, 503 service unavailable, 504 gateway timeout, and clients retrying repeatedly.

## Likely causes
A dependency that is slow or down, resource exhaustion in that dependency such as a full connection pool or CPU saturation, a network problem, a deploy in progress, or timeouts that do not line up between layers.

## Diagnosis
Find the first failing dependency by looking for the earliest error timestamp, not the loudest message. Check health endpoints and trace the request path. Compare timeout settings across layers.

## Fix
Restart or scale the failing dependency, fail over if possible, and add retries with exponential backoff and a circuit breaker so that clients do not create a retry storm.

## Prevention
Use circuit breakers, keep outer timeouts longer than inner ones, and add dependency health checks.
