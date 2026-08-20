# EZ0030 File\_Operation\_Error\_Open

## 错误信息

报错格式如下，占位符%s的含义依次为文件路径、报错原因：

```text
Failed to open file %s. Reason: %s.
```

报错示例如下：

```text
Failed to open file /home/ops-cv/build/tests/ut/op_host/aipp_ut_test_NA.cfg. Reason: [Errno 2] No such file or directory.
```

## 解决方法

1. 正确配置文件路径。

2. 正确配置文件权限。

3. 重新安装软件包。
