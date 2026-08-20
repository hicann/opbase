# EZ0015 Invalid\_Argument\_Tensor\_Shape\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape大小错误值、报错原因：

```text
Parameter %s of %s has incorrect shape size %s. Reason: %s.
```

报错示例如下：

```text
Parameter y of ResizeLinear has incorrect shape size [144,144,1]. Reason: The linear-dimension of output y must be equal to value (256) of input parameter size.
```

## 解决方法

根据报错原因检查输入或输出tensor的shape大小是否满足条件。
