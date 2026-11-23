# Reading logs during an incident

## Levels and time
INFO describes normal activity, WARN means something unusual that may need attention, and ERROR means an operation failed. Always check the time zone and correlate events across components by timestamp.

## Finding the start
Look for the first error, not the loudest one. Read the lines just before it, because warnings such as slow queries often come first. Group repeated messages into patterns by ignoring numbers and identifiers, and count each pattern.

## Stack traces
A stack trace belongs to the log line above it. Read from the top to find the failing call and the original cause.
