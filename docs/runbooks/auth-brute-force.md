# Authentication brute force and credential stuffing

## Symptoms
Many "failed login for user" warnings in a short time, "account locked after repeated failures", rate limit warnings, and attempts coming from a small set of IP addresses or ranges.

## Likely causes
Credential stuffing with leaked passwords, brute-force password guessing, or a misconfigured client stuck in a retry loop with a wrong password.

## Diagnosis
Count failures per source address and per user. Check whether many different users are tried from the same address, which suggests stuffing. Look for any successful login that follows a long run of failures, because that may mean an account is compromised.

## Fix
Block or rate-limit the offending addresses, force a password reset for locked or possibly compromised accounts, and turn on multi-factor authentication or a CAPTCHA for the login endpoint.

## Prevention
Use lockout and backoff policies, check passwords against known breach lists, and alert on spikes in failed logins.
