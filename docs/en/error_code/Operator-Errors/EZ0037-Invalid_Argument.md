# EZ0037 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, error cause.

```text
Parameter %s of %s is invalid. Reason: %s.
```

Error example:

```text
Parameter blockTable of IncreFlashAttention is invalid. Reason: blockTable must be empty when D of query and key is not equal to D of value.
```

## Solution

Check whether the input/output tensor meets the condition.
