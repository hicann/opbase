# EZ0020 Invalid\_Argument\_Tensor\_Dtype

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, incorrect data type, error cause.

```text
Parameter %s of %s has incorrect dtype %s. Reason: %s.
```

Error example:

```text
Parameter k_cache of KvRmsNormRopeCache has incorrect dtype FLOAT16. Reason: The dtype of input k_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2, or the same as the dtype of input kv.
```

## Solution

Check whether the dtype of input/output tensor meets the condition.
