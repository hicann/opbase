# EZ0010 Invalid\_Argument\_Tensor\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape错误值、报错原因：

```text
Parameters %s of %s have incorrect shapes %s. Reason: %s.
```

报错示例如下：

```text
Parameters indices 0th tensor and x 0th tensor of DynamicStitch have incorrect shapes [2,3] and [4,5,6]. Reason: The shape of indices's tensor should match the shape formed by the first 2 axes of x's corresponding tensor.
```

## 解决方法

根据报错原因检查输入或输出tensor的shape是否满足要求。
