# EZ0012 Invalid\_Argument\_Tensor\_Shape\_Dim

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape dim错误值、报错原因：

```text
Parameter %s of %s has incorrect shape dim %s. Reason: %s.
```

报错示例如下：

```text
Parameter query of ApplyRotaryPosEmb has incorrect shape dim 3D. Reason: The shape dims of input query must be 4 when the attr layout is 1 (BSND).
```

## 解决方法

根据报错原因检查输入或输出tensor的shape dim是否正确。
