# EZ0024 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、错误值、正确值：

```text
Parameter %s of %s has incorrect value %s. It should be %s.
```

报错示例1如下：

```text
Parameter update_type of AttentionUpdate has incorrect value 2. It should be 0 or 1.
```

报错示例2如下：

```text
Parameter sp of operator AttentionUpdate has incorrect value 17. It should be in range of [1, 16].
```

## 解决方法

检查参数值是否正确。
