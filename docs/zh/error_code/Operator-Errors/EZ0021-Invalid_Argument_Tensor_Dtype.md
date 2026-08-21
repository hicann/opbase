# EZ0021 Invalid\_Argument\_Tensor\_Dtype

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、数据类型错误值、报错原因：

```text
Parameters %s of %s have incorrect dtypes %s. Reason: %s.
```

报错示例如下：

```text
Parameters cos and dy of RotaryPositionEmbeddingGrad have incorrect dtypes FLOAT16 and FLOAT32. Reason: The dtypes of input cos and input dy should be the same.
```

## 解决方法

根据报错原因检查输入或输出tensor的数据类型是否满足条件。
