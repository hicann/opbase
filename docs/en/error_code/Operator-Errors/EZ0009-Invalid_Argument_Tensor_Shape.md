# EZ0009 Invalid\_Argument\_Tensor\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape, error cause.

```text
Parameter %s of %s has incorrect shape [%s]. Reason: %s.
```

Error example:

```text
Parameter indices 0th tensor of DynamicStitch has incorrect shape [2,-1,128]. Reason: The input indices's tensor has negative dimension.
```

## Solution

Check whether the shape of input/output tensor is correct.
