# EZ0023 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameters name, operator name or API name, invalid tensor nums, error cause.

```text
Parameters %s of %s have invalid tensor nums %s. Reason: %s.
```

Error example:

```text
Parameters lse and go of AttentionUpdate have invalid tensor nums 4 and 4. Reason: The number of tensors in input lse and go should be twice the attr sp, where sp is 2.
```

## Solution

Check whether the number of tensors in the input tensor lists meets the conditions.
