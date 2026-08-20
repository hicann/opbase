# EZ0034 Config\_Error

## 错误信息

报错格式如下，占位符%s的含义依次为配置值、配置项、配置文件、报错原因：

```text
Values %s of configuration items %s in configuration file %s are invalid. Reason: %s.
```

报错示例如下：

```text
Values 256, 221 of configuration items src_image_size_h, src_image_size_w in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_28.cfg are invalid. Reason: When input_format is YUV420SP_U8, src_image_size_h and src_image_size_w must be even numbers.
```

## 解决方法

参考用户指南中的规格修改配置文件。
