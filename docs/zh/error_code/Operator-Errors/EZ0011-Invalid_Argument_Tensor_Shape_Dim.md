# EZ0011 Invalid\_Argument\_Tensor\_Shape\_Dim

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape dim错误值、shape dim正确值：

```text
Parameter %s of %s has incorrect shape dim %s. It should be %s.
```

报错示例1如下：

```text
Parameter array of Bincount has incorrect shape dim 2D. It should be 1D.
```

报错示例2如下：

```text
Parameter x of SoftmaxV2 has incorrect shape dim 5D. It should be less than or equal to 4.
```

## 解决方法

检查输入或输出tensor的shape dim是否正确。
