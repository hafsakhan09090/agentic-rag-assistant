# Database connection pool exhaustion

## Symptoms
Errors such as "connection pool exhausted", requests waiting in a queue, "wait time exceeded threshold", API calls timing out and the gateway returning 503. Active connections sit at the maximum, for example 50 of 50.

## Likely causes
Connections that are never returned to the pool (leaks), long-running queries or transactions holding connections, a traffic spike, a pool that is too small, or a slow database that makes every query hold its connection longer.

## Diagnosis
Compare active connections with the configured maximum. List the longest running queries and any sessions that are idle in transaction. Check whether the problem started after a deploy. Watch whether active connections ever drop; if they never come back down, suspect a leak.

## Fix
Terminate stuck queries, restart the service that leaks connections, temporarily raise the pool size, and shed load while the cause is fixed. Set a statement timeout so one slow query cannot hold a connection forever.

## Prevention
Use context managers so connections are always released, set connection and acquire timeouts, and alert when pool usage passes 80 percent.
