# EZ0017 Invalid\_Argument\_Tensor\_Format

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、format错误值、format正确值：

```text
Parameter %s of %s has incorrect format %s. It should be %s.
```

报错示例如下：

```text
Parameter x of ResizeBilinearV2 has incorrect format ND. It should be NCHW or NHWC.
```

## 解决方法

检查输入或输出tensor的format是否正确。
