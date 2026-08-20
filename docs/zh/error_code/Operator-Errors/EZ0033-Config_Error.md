# EZ0033 Config\_Error

## 错误信息

报错格式如下，占位符%s的含义依次为配置值、配置项、配置文件、报错原因：

```text
Value %s of configuration item %s in configuration file %s is invalid. Reason: %s.
```

报错示例如下：

```text
Value true of configuration item ax_swap_switch in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_05.cfg is invalid. Reason: If the format of the input images is not XRGB888_U8, the value of this configuration item must be 'false'.
```

## 解决方法

参考用户指南中的规格修改配置文件。
