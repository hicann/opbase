# EZ0012 Invalid\_Argument\_Tensor\_Shape\_Dim

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape dim, error cause.

```text
Parameter %s of %s has incorrect shape dim %s. Reason: %s.
```

Error example:

```text
Parameter query of ApplyRotaryPosEmb has incorrect shape dim 3D. Reason: The shape dims of input query must be 4 when the attr layout is 1 (BSND).
```

## Solution

Check whether the shape dimension of input/output tensor is correct.
