# EZ1010 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, parameter name, error cause.

```text
Value %s for parameter %s is invalid. Reason: %s.
```

Error example:

```text
Value 0 for parameter workspaceLen is invalid. Reason: The passed workspace size 0 does not meet the workspace size 1024 actually required by the operator.
```

## Solution

Please adjust the parameter value as prompted in the Reason.
