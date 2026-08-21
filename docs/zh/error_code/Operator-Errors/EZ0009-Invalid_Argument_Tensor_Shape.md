# EZ0009 Invalid\_Argument\_Tensor\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape错误值、报错原因：

```text
Parameter %s of %s has incorrect shape [%s]. Reason: %s.
```

报错示例如下：

```text
Parameter indices 0th tensor of DynamicStitch has incorrect shape [2,-1,128]. Reason: The input indices's tensor has negative dimension.
```

## 解决方法

根据报错原因检查输入或输出tensor的shape是否正确。
