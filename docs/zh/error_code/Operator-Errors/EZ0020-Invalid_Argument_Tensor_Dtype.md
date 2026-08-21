# EZ0020 Invalid\_Argument\_Tensor\_Dtype

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、数据类型错误值、报错原因：

```text
Parameter %s of %s has incorrect dtype %s. Reason: %s.
```

报错示例如下：

```text
Parameter k_cache of KvRmsNormRopeCache has incorrect dtype FLOAT16. Reason: The dtype of input k_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2, or the same as the dtype of input kv.
```

## 解决方法

根据报错原因检查输入或输出tensor的数据类型是否满足条件。
