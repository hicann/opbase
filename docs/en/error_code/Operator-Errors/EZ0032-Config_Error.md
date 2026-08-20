# EZ0032 Config\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: configuration item value, configuration item name, file path, correct value.

```text
Value %s of configuration item %s in configuration file %s is invalid, it should be %s.
```

Error example:

```text
Value NA of configuration item aipp_mode in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_27.cfg is invalid, it should be static or dynamic.
```

## Solution

Modify the configuration file by referring to the specifications in the user guide.
