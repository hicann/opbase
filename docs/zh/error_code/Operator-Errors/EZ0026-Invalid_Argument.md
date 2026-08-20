# EZ0026 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、错误值、报错原因：

```text
Parameter %s of %s has incorrect value %s. Reason: %s.
```

报错示例如下：

```text
Parameter output_size of UpsampleNearest3D has incorrect value (2147483648, 512, 512). Reason: Each value of output_size must be less than or equal to INT32_MAX.
```

## 解决方法

根据报错原因检查参数值是否正确。
