# Disk full

## Symptoms
"No space left on device", write failures, checkpoint or temp file errors, jobs failing with IOError, disk usage above 95 percent, or "inode usage high".

## Likely causes
Logs growing without rotation, temporary files that were never cleaned, large uploads, backup artifacts, or files that were deleted but are still held open by a running process.

## Diagnosis
Run df -h for space and df -i for inodes. Find the biggest directories with du -sh on each top-level path, sorted by size. Use lsof with +L1 to find deleted files still held open. Inode exhaustion can happen even when df shows free space, usually because of millions of tiny files.

## Fix
Rotate and compress logs, remove temporary files, expand the volume, and restart the process holding deleted files. For inode exhaustion, delete or archive the directory with many small files.

## Prevention
Configure logrotate, alert at 80 and 90 percent usage, set quotas, and clean temp directories on a schedule.
