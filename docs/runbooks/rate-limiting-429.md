# Rate limiting and HTTP 429

## Symptoms
"rate limit exceeded", HTTP 429 responses, bursts of requests from a single IP address or API key, and legitimate clients being blocked.

## Likely causes
Abusive clients, retry storms from clients without backoff, limits configured too low, or many users behind one shared address.

## Diagnosis
List the top clients by request count, check the limit configuration, and look for retry loops.

## Fix
Adjust limits per client, tell clients to back off using a Retry-After header, and block clearly abusive sources.

## Prevention
Use tiered limits, document retry behaviour, and monitor 429 rates.
