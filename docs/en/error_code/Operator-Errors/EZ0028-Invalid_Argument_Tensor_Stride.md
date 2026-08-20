# EZ0028 Invalid\_Argument\_Tensor\_Stride

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect stride, correct stride.

```text
Parameter %s of %s has incorrect stride %s, it should be %s.
```

Error example:

```text
Parameter x of Conv2dv2 has incorrect stride [1, 2, 3, 4], it should be [1, 1, 1, 1].
```

## Solution

Check whether the stride of input/output tensor is correct.
