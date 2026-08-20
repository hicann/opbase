# EZ1007 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: shape, operator name, incorrect dimension, error cause.

```text
Shape %s of the tensor of operator %s has incorrect dimension %s. Reason: %s.
```

Error example:

```text
Shape [4, 2] of the tensor of operator aclnnAdd_0 has incorrect dimension 2. Reason: The tensor whose shape is [4, 2] and the tensor whose shape is [4, 3] do not meet the broadcast condition.
```

## Solution

Please locate the issue as prompted in the Reason and provide the correct shape.
