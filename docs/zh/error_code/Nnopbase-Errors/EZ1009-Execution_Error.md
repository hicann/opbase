# EZ1009 Execution\_Error

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、报错原因：

```text
Failed to execute operator %s. Reason: %s.
```

报错示例如下：

```text
Failed to execute operator AddCustom. Reason: The dtype or format of the actual input or output parameter of the operator is inconsistent with that defined in the operator prototype OpDef.
```

## 解决方法

根据Reason中的提示定位问题。
