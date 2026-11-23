# TLS certificate expiry

## Symptoms
"TLS handshake failed", "certificate has expired", "SSL certificate verify failed", errors connecting to an upstream such as a payments provider, and browser security warnings.

## Likely causes
The automatic renewal job failed, the new certificate was not installed on every node, the intermediate chain is incomplete, or the server clock is wrong.

## Diagnosis
Check the dates with openssl s_client connecting to the host, piped into openssl x509 with the dates option. Verify the full chain is served. Look at the renewal job logs. Compare the server time with a trusted time source, because a wrong clock makes valid certificates look expired.

## Fix
Renew and install the certificate, reload the service, deploy the full chain, and fix time synchronisation with NTP if the clock is off.

## Prevention
Monitor expiry dates and alert at 30, 14 and 7 days, automate renewal, and keep an inventory of every certificate and where it is used.
