# EZ1005 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, error cause.

```text
Failed to parse file %s. Reason: %s.
```

Error example:

```text
Failed to parse file /home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/config/ascendxxx/aic-ascendxxxx-ops-info-oam.json. Reason: The operator JSON file is not in the standard key-value structure.
```

## Possible Cause

1.The custom operator JSON file is damaged.

2.The built-in operator JSON file is damaged.

## Solution

1.Reinstall the custom operator package.

2.Reinstall the built-in operator package.
