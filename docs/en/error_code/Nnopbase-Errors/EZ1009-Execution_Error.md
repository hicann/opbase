# EZ1009 Execution\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, error cause.

```text
Failed to execute operator %s. Reason: %s.
```

Error example:

```text
Failed to execute operator AddCustom. Reason: The dtype or format of the actual input or output parameter of the operator is inconsistent with that defined in the operator prototype OpDef.
```

## Solution

Please locate the issue as prompted in the Reason.
