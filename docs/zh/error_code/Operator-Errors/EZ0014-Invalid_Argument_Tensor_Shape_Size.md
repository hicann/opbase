# EZ0014 Invalid\_Argument\_Tensor\_Shape\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、shape大小错误值、shape大小正确值：

```text
Parameter %s of %s has incorrect shape size %s. It should be %s.
```

报错示例如下：

```text
Parameter group_index of SwigluMxQuantWithDualAxis has incorrect shape size 0. It should be > 0.
```

## 解决方法

检查输入或输出tensor的shape大小是否正确。
