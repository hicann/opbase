# EZ1011 Invalid\_Argument\_Null\_Pointer

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: API name, parameter name, error cause.

```text
%s failed because %s cannot be a NULL pointer.
```

Error example:

```text
NnopbaseRunWithWorkspace failed because executor cannot be a NULL pointer.
```

## Solution

Please adjust the parameter value as prompted in the error message.
