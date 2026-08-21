# EZ0019 Invalid\_Argument\_Tensor\_Dtype

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect data type, correct data type.

```text
Parameter %s of %s has incorrect dtype %s. It should be %s.
```

Error example:

```text
Parameter softmax of SoftmaxGrad has incorrect dtype INT32. It should be FLOAT, FLOAT16 or BF16.
```

## Solution

Check whether the dtype of input/output tensor is correct.
