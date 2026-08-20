# EZ0004 Invalid\_Input

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name.

```text
Parameter %s of %s is required, but it is empty.
```

Error example:

```text
Parameter input of Cumsum is required, but it is empty.
```

## Solution

Check whether the required parameters of the operator are correctly set.
