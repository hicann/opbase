# EZ0023 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、tensor数量错误值、报错原因：

```text
Parameters %s of %s have invalid tensor nums %s. Reason: %s.
```

报错示例如下：

```text
Parameters lse and go of AttentionUpdate have invalid tensor nums 4 and 4. Reason: The number of tensors in input lse and go should be twice the attr sp, where sp is 2.
```

## 解决方法

根据报错原因检查输入或输出tensor的数量是否满足条件。
