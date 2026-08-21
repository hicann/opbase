# EZ0015 Invalid\_Argument\_Tensor\_Shape\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape size, error cause.

```text
Parameter %s of %s has incorrect shape size %s. Reason: %s.
```

Error example:

```text
Parameter y of ResizeLinear has incorrect shape size [144,144,1]. Reason: The linear-dimension of output y must be equal to value (256) of input parameter size.
```

## Solution

Check whether the shape size of input/output tensor meets the condition.
