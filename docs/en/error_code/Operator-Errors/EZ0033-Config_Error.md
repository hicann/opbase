# EZ0033 Config\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: configuration item value, configuration item name, file path, error cause.

```text
Value %s of configuration item %s in configuration file %s is invalid. Reason: %s.
```

Error example:

```text
Value true of configuration item ax_swap_switch in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_05.cfg is invalid. Reason: If the format of the input images is not XRGB888_U8, the value of this configuration item must be 'false'.
```

## Solution

Modify the configuration file by referring to the specifications in the user guide.
