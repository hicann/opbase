# EZ0008 Invalid\_Argument\_Tensor\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape错误值、shape正确值：

```text
Parameter %s of %s has incorrect shape [%s]. It should be [%s].
```

报错示例如下：

```text
Parameter y of Ger has incorrect shape [128]. It should be [32,64].
```

## 解决方法

检查输入或输出tensor的shape是否正确。
