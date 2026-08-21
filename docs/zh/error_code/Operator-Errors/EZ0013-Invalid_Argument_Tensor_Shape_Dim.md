# EZ0013 Invalid\_Argument\_Tensor\_Shape\_Dim

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape dim错误值、报错原因：

```text
Parameters %s of %s have incorrect shape dims %s. Reason: %s.
```

报错示例如下：

```text
Parameters dy, cos and sin of RotaryPositionEmbeddingGrad have incorrect shape dims 3, 4 and 4. Reason: The numbers of dimensions of input dy, cos and sin should all be 3D or 4D.
```

## 解决方法

根据报错原因检查输入或输出tensor的shape维度值是否满足要求。
