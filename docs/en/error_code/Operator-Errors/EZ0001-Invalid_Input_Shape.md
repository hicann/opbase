# EZ0001 Invalid\_Input\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: index, operator name or API name, incorrect shape, correct shape.

```text
The %sth input of %s has incorrect shape [%s]. It should be [%s].
```

Error example:

```text
The 0th input of MatMulV2 has incorrect shape [2,128,256]. It should be [M, K].
```

## Solution

Check whether the shape of the input tensor is correct.
