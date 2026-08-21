# EZ0025 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、列表大小错误值、列表大小正确值：

```text
Parameter %s of %s has invalid list size %s. It should be %s.
```

报错示例如下：

```text
Parameter axes of SoftmaxV2 has incorrect element nums 2. It should be 1.
```

## 解决方法

检查参数列表大小是否正确。
