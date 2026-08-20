# EZ0031 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file path, error cause.

```text
Failed to parse file %s. Reason: %s.
```

Error example:

```text
Failed to parse file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_26.cfg. Reason: The AIPP operator configuration file does not contain configuration item aipp_mode.
```

## Solution

Modify the configuration file by referring to the specifications in the user guide.
