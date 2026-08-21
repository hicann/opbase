# EZ0026 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect value, error cause.

```text
Parameter %s of %s has incorrect value %s. Reason: %s.
```

Error example:

```text
Parameter output_size of UpsampleNearest3D has incorrect value (2147483648, 512, 512). Reason: Each value of output_size must be less than or equal to INT32_MAX.
```

## Solution

Check whether the parameter value is correct.
