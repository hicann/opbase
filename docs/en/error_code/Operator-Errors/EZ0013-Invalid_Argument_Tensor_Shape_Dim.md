# EZ0013 Invalid\_Argument\_Tensor\_Shape\_Dim

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect shape dims, error cause.

```text
Parameters %s of %s have incorrect shape dims %s. Reason: %s.
```

Error example:

```text
Parameters dy, cos and sin of RotaryPositionEmbeddingGrad have incorrect shape dims 3, 4 and 4. Reason: The numbers of dimensions of input dy, cos and sin should all be 3D or 4D.
```

## Solution

Check whether the shape dimensions of input/output tensors meet the relationship.
