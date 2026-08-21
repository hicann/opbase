# EZ0008 Invalid\_Argument\_Tensor\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape, correct shape.

```text
Parameter %s of %s has incorrect shape [%s]. It should be [%s].
```

Error example:

```text
Parameter y of Ger has incorrect shape [128]. It should be [32,64].
```

## Solution

Check whether the shape of input/output tensor is correct.
