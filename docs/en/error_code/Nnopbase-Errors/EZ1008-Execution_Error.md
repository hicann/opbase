# EZ1008 Execution\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, error cause.

```text
Failed to execute operator %s. Reason: %s.
```

Error example:

```text
Failed to execute operator aclnnAdd_0_AddAiCore. Reason: The infershape function does not exist.
```

## Solution

1.If the tiling function does not exist, check whether the tiling function is registered successfully.

2.If the tiling function fails to be executed, check the implementation logic of the tiling function.
