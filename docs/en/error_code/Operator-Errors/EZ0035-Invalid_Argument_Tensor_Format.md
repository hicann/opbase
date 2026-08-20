# EZ0035 Invalid\_Argument\_Tensor\_Format

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect format, error cause.

```text
Parameter %s of %s has incorrect format %s. Reason: %s.
```

Error example:

```text
Parameter filter of aclnnConvolutionGetWorkspaceSize has incorrect format FRACTAL_Z_C04. Reason: The value of this parameter can be FRACTAL_Z_C04 only when the SoC version is Ascend950.
```

## Solution

Check whether the format of input/output tensor meet the condition.
