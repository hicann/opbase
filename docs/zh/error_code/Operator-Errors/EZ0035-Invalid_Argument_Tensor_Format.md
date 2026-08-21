# EZ0035 Invalid\_Argument\_Tensor\_Format

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、错误format、报错原因：

```text
Parameter %s of %s has incorrect format %s. Reason: %s.
```

报错示例如下：

```text
Parameter filter of aclnnConvolutionGetWorkspaceSize has incorrect format FRACTAL_Z_C04. Reason: The value of this parameter can be FRACTAL_Z_C04 only when the SoC version is Ascend950.
```

## 解决方法

检查输入/输出tensor的format是否符合要求。
