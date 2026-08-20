# EZ1007 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为输入参数Shape值、算子名、dim值、报错原因：

```text
Shape %s of the tensor of operator %s has incorrect dimension %s. Reason: %s.
```

报错示例如下：

```text
Shape [4, 2] of the tensor of operator aclnnAdd_0 has incorrect dimension 2. Reason: The tensor whose shape is [4, 2] and the tensor whose shape is [4, 3] do not meet the broadcast condition.
```

## 解决方法

需按照Reason中的提示定位问题，提供正确的输入。
