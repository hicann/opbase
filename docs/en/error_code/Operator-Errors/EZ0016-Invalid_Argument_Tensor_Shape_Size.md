# EZ0016 Invalid\_Argument\_Tensor\_Shape\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect shape sizes, error cause.

```text
Parameters %s of %s have incorrect shape sizes %s. Reason: %s.
```

Error example 1:

```text
Parameters x and y of SwigluMxQuantWithDualAxis have incorrect shape sizes 1024 and 0. Reason: The shape size of x must be equal to the shape size of y.
```

Error example 2:

```text
Parameters query, key, cos and sin of operator ApplyRotaryPosEmb have incorrect shape sizes 0, 0, 0 and 0. Reason: All inputs must be non-empty tensors.
```

## Solution

Check whether the shape sizes of input/output tensors meet the relationship.
