# EZ0018 Invalid\_Argument\_Tensor\_Format

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、format错误值、报错原因：

```text
Parameters %s of %s have incorrect formats %s. Reason: %s.
```

报错示例如下：

```text
Parameters x and y of OperatorName have incorrect formats ND and NCHW. Reason: The formats of all inputs must match.
```

## 解决方法

根据报错原因检查输入或输出tensor的format是否满足条件。
