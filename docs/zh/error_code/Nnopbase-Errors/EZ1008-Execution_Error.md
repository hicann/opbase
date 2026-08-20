# EZ1008 Execution\_Error

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、报错原因：

```text
Failed to execute operator %s. Reason: %s.
```

报错示例如下：

```text
Failed to execute operator aclnnAdd_0_AddAiCore. Reason: The infershape function does not exist.
```

## 解决方法

1. 如果tiling函数不存在，请检查tiling函数是否已成功注册。
2. 如果tiling函数执行失败，请检查tiling函数的实现逻辑。
