# DNS resolution failures

## Symptoms
"could not resolve host", "name or service not known", intermittent connection failures, and timeouts on the first request to a host.

## Likely causes
A DNS server outage, a bad resolver configuration, an expired or wrong record, or stale cached answers.

## Diagnosis
Query the name with dig or nslookup against different resolvers, check the resolver configuration file, and compare answers.

## Fix
Switch to a healthy resolver, correct the record, and flush caches.

## Prevention
Configure redundant resolvers and use local caching.
