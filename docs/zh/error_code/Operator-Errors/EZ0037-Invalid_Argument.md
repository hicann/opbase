# EZ0037 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、报错原因：

```text
Parameter %s of %s is invalid. Reason: %s.
```

报错示例如下：

```text
Parameter blockTable of IncreFlashAttention is invalid. Reason: blockTable must be empty when D of query and key is not equal to D of value.
```

## 解决方法

检查输入/输出tensor是否符合要求。
