# EZ0034 Config\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: configuration items value, configuration items name, file path, error cause.

```text
Values %s of configuration items %s in configuration file %s are invalid. Reason: %s.
```

Error example:

```text
Values 256, 221 of configuration items src_image_size_h, src_image_size_w in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_28.cfg are invalid. Reason: When input_format is YUV420SP_U8, src_image_size_h and src_image_size_w must be even numbers.
```

## Solution

Modify the configuration file by referring to the specifications in the user guide.
