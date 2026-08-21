# EZ1014 Execution\_Error

## 错误信息

报错格式如下，占位符%s分别表示参数名、报错原因：

```text
Failed to execute operator %s. Reason: %s.
```

报错示例如下：

```text
Failed to execute operator aclnnFlashAttentionVarLenScore_0_FlashAttentionScore. Reason: Failed to execute inferShape.
```

## 解决方法

1. 如果inferShape函数不存在，请检查inferShape函数是否已成功注册。
2. 如果inferShape函数执行失败，请检查inferShape函数的实现逻辑。
