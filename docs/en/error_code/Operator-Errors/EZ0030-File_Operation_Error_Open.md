# EZ0030 File\_Operation\_Error\_Open

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file path, error cause.

```text
Failed to open file %s. Reason: %s.
```

Error example:

```text
Failed to open file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_NA.cfg. Reason: [Errno 2] No such file or directory.
```

## Solution

1. Configure the file path correctly.

2. Configure the file permissions correctly.

3. Reinstall the package.
