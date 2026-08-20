# EZ1010 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名、报错原因：

```text
Value %s for parameter %s is invalid. Reason: %s.
```

报错示例如下：

```text
Value 0 for parameter workspaceLen is invalid. Reason: The passed workspace size 0 does not meet the workspace size 1024 actually required by the operator.
```

## 解决方法

根据Reason中的提示调整参数值。
