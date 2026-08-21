# EZ0016 Invalid\_Argument\_Tensor\_Shape\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape大小错误值、报错原因：

```text
Parameters %s of %s have incorrect shape sizes %s. Reason: %s.
```

报错示例1如下：

```text
Parameters x and y of SwigluMxQuantWithDualAxis have incorrect shape sizes 1024 and 0. Reason: The shape size of x must be equal to the shape size of y.
```

报错示例2如下：

```text
Parameters query, key, cos and sin of operator ApplyRotaryPosEmb have incorrect shape sizes 0, 0, 0 and 0. Reason: All inputs must be non-empty tensors.
```

## 解决方法

根据报错原因检查输入或输出tensor的shape大小是否满足条件。
