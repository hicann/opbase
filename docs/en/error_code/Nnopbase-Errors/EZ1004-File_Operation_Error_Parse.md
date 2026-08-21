# EZ1004 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, error cause.

```text
Failed to parse file %s. Reason: %s.
```

Error example:

```text
Failed to parse file /home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe//kernel/ascendxxxx/ops_legacy/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.json. Reason: [json.exception.parse_error.101] parse error at line 4, column 14: syntax error while parsing object - unexpected string literal; expected '}'.
```

## Solution

Please locate the issue as prompted in the Reason and provide the correct file.
