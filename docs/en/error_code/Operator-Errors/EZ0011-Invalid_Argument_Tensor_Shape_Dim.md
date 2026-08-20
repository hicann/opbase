# EZ0011 Invalid\_Argument\_Tensor\_Shape\_Dim

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect shape dim, correct shape dim.

```text
Parameter %s of %s has incorrect shape dim %s. It should be %s.
```

Error example 1:

```text
Parameter array of Bincount has incorrect shape dim 2D. It should be 1D.
```

Error example 2:

```text
Parameter x of SoftmaxV2 has incorrect shape dim 5D. It should be less than or equal to 4.
```

## Solution

Check whether the shape dimension of input/output tensor is correct.
