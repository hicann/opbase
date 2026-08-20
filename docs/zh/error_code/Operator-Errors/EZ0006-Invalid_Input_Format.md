# EZ0006 Invalid\_Input\_Format

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、format错误值、format正确值：

```text
Input parameter %s of %s has incorrect format %s. It should be %s.
```

报错示例如下：

```text
Input parameter x of BatchNorm has incorrect format NHWC. It should be NCHW. 
```

## 解决方法

请按照报错提示修改参数值。
