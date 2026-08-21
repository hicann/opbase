# EZ0032 Config\_Error

## 错误信息

报错格式如下，占位符%s的含义依次为配置值、配置项、配置文件、正确值：

```text
Value %s of configuration item %s in configuration file %s is invalid, it should be %s.
```

报错示例如下：

```text
Value NA of configuration item aipp_mode in configuration file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_27.cfg is invalid, it should be static or dynamic.
```

## 解决方法

参考用户指南中的规格修改配置文件。
