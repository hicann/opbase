# EZ1014 Execution\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, error cause.

```text
Failed to execute operator %s. Reason: %s.
```

Error example:

```text
Failed to execute operator aclnnFlashAttentionVarLenScore_0_FlashAttentionScore. Reason: Failed to execute inferShape.
```

## Solution

1.If the inferShape function does not exist, check whether the inferShape function is registered successfully.

2.If the inferShape function fails to be executed, check the implementation logic of the inferShape function.
