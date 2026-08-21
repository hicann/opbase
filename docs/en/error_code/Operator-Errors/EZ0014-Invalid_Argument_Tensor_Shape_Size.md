# EZ0014 Invalid\_Argument\_Tensor\_Shape\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape size, correct shape size.

```text
Parameter %s of %s has incorrect shape size %s. It should be %s.
```

Error example:

```text
Parameter group_index of SwigluMxQuantWithDualAxis has incorrect shape size 0. It should be > 0.
```

## Solution

Check whether the shape size of input/output tensor is correct.
