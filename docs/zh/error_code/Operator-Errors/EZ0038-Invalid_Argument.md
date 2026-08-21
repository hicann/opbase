# EZ0038 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、错误元素数量、报错原因：

```text
Parameter %s of %s has incorrect element nums %s. Reason: %s.
```

报错示例如下：

```text
Parameter actualSequenceLengthKV of FusedInferAttentionScore has incorrect element nums 0. Reason: 
When layout is TND, actualSequenceLengthKV must be input and contain one or more elements.
```

## 解决方法

检查列表大小是否正确。
