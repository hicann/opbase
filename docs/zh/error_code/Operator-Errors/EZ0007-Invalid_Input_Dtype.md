# EZ0007 Invalid\_Input\_Dtype

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、数据类型错误值、数据类型正确值：

```text
Input parameter %s of %s has incorrect dtype %s. It should be %s.
```

报错示例如下：

```text
Input parameter sink of aclnnFlashAttentionScore has incorrect dtype INT32. It should be FLOAT. 
```

## 解决方法

请按照报错提示修改参数值。
