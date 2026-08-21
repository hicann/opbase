# EZ0010 Invalid\_Argument\_Tensor\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect shapes, error cause.

```text
Parameters %s of %s have incorrect shapes %s. Reason: %s.
```

Error example:

```text
Parameters indices 0th tensor and x 0th tensor of DynamicStitch have incorrect shapes [2,3] and [4,5,6]. Reason: The shape of indices's tensor should match the shape formed by the first 2 axes of x's corresponding tensor.
```

## Solution

Check whether the shapes of input/output tensors meet the relationship.
