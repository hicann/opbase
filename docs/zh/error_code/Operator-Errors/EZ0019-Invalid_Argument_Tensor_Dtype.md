# EZ0019 Invalid\_Argument\_Tensor\_Dtype

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、数据类型错误值、数据类型正确性：

```text
Parameter %s of %s has incorrect dtype %s. It should be %s.
```

报错示例如下：

```text
Parameter softmax of SoftmaxGrad has incorrect dtype INT32. It should be FLOAT, FLOAT16 or BF16.
```

## 解决方法

检查输入或输出tensor的数据类型是否正确。
