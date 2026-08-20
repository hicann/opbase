# EZ0001 Invalid\_Input\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为序号、算子名或接口名、shape错误值、shape正确值：

```text
The %sth input of %s has incorrect shape [%s]. It should be [%s].
```

报错示例如下：

```text
The 0th input of MatMulV2 has incorrect shape [2,128,256]. It should be [M, K].
```

## 解决方法

检查输入tensor的shape是否正确。
