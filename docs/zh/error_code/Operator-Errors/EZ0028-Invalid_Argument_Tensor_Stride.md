# EZ0028 Invalid\_Argument\_Tensor\_Stride

## 错误信息

报错格式如下，占位符%s的含义依次为输入参数名、算子名或接口名、stride错误值、stride正确值：

```text
Parameter %s of %s has incorrect stride %s, it should be %s.
```

报错示例如下：

```text
Parameter x of Conv2dv2 has incorrect stride [1, 2, 3, 4], it should be [1, 1, 1, 1].
```

## 解决方法

检查输入/输出tensor的stride是否正确。
