# EZ0027 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, incorrect values, error cause.

```text
Parameters %s of %s have incorrect values %s. Reason: %s.
```

Error example:

```text
Parameters align_corners and half_pixel_centers of ResizeNearestNeighborV2Grad have incorrect values true and true. Reason: The values of attributes align_corners and half_pixel_centers cannot be true at the same time.
```

## Solution

Check whether the parameter values meet the condition.
