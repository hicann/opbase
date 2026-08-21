# EZ1011 Invalid\_Argument\_Null\_Pointer

## 错误信息

报错格式如下，占位符%s的含义依次为接口名、参数名：

```text
%s failed because %s cannot be a NULL pointer.
```

报错示例如下：

```text
NnopbaseRunWithWorkspace failed because executor cannot be a NULL pointer.
```

## 解决方法

根据报错提示调整参数值。
