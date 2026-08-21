# EZ0038 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect element nums, error cause.

```text
Parameter %s of %s has incorrect element nums %s. Reason: %s.
```

Error example:

```text
Parameter actualSequenceLengthKV of FusedInferAttentionScore has incorrect element nums 0. Reason: 
When layout is TND, actualSequenceLengthKV must be input and contain one or more elements.
```

## Solution

Check whether the list size is correct.
