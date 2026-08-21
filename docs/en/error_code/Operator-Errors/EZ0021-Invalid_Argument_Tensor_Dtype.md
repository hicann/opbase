# EZ0021 Invalid\_Argument\_Tensor\_Dtype

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect data types, error cause.

```text
Parameters %s of %s have incorrect dtypes %s. Reason: %s.
```

Error example:

```text
Parameters cos and dy of RotaryPositionEmbeddingGrad have incorrect dtypes FLOAT16 and FLOAT32. Reason: The dtypes of input cos and input dy should be the same.
```

## Solution

Check whether the dtypes of input/output tensors meet the condition.
