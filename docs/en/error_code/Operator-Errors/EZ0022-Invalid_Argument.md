# EZ0022 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, operator name or API name, invalid tensor num, correct tensor num.

```text
Parameter %s of %s has invalid tensor num %ld. It should be %s.
```

Error example:

```text
Parameter instance of Foreach has invalid tensor num 1000. It should be within the range [1, 950].
```

## Solution

Check whether the number of tensor in the input tensor list meets the condition.
