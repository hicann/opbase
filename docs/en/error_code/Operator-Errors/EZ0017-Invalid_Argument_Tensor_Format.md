# EZ0017 Invalid\_Argument\_Tensor\_Format

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect format, correct format.

```text
Parameter %s of %s has incorrect format %s. It should be %s.
```

Error example:

```text
Parameter x of ResizeBilinearV2 has incorrect format ND. It should be NCHW or NHWC.
```

## Solution

Check whether the format of input/output tensor is correct.
