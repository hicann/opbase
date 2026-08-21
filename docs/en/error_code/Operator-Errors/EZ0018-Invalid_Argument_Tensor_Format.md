# EZ0018 Invalid\_Argument\_Tensor\_Format

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect formats, error cause.

```text
Parameters %s of %s have incorrect formats %s. Reason: %s.
```

Error example:

```text
Parameters x and y of OperatorName have incorrect formats ND and NCHW. Reason: The formats of all inputs must match.
```

## Solution

Check whether the formats of input/output tensors meet the condition.
