# EZ0031 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s的含义依次为文件路径、报错原因：

```text
Failed to parse file %s. Reason: %s.
```

报错示例如下：

```text
Failed to parse file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_26.cfg. Reason: The AIPP operator configuration file does not contain configuration item aipp_mode.
```

## 解决方法

参考用户指南中的规格修改配置文件。
