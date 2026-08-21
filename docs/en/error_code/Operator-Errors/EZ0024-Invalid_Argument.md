# EZ0024 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect value, correct value.

```text
Parameter %s of %s has incorrect value %s. It should be %s.
```

Error example 1:

```text
Parameter update_type of AttentionUpdate has incorrect value 2. It should be 0 or 1.
```

Error example 2:

```text
Parameter sp of operator AttentionUpdate has incorrect value 17. It should be in range of [1, 16].
```

## Solution

Check whether the parameter value is correct.
