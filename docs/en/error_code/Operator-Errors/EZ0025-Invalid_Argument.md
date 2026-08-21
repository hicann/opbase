# EZ0025 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, invalid list size, correct list size.

```text
Parameter %s of %s has invalid list size %s. It should be %s.
```

Error example:

```text
Parameter axes of SoftmaxV2 has incorrect element nums 2. It should be 1.
```

## Solution

Check whether the parameter list size is correct.
