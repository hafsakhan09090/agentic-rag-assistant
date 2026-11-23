# Deployment problems and rollback

## Symptoms
Errors that start right after a deploy, new error patterns that were never seen before, or a version mismatch between services.

## Likely causes
A bad configuration, an incompatible database migration, or a missing environment variable.

## Diagnosis
Compare the deploy time with the time the errors began. Diff the configuration between versions. Check canary metrics if they exist.

## Fix
Roll back to the last known good version first, and investigate afterwards. Turn off the feature flag if the change was behind one.

## Prevention
Use canary or staged rollouts and automatic rollback when the error rate rises.
