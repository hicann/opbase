# EZ1006 Not\_Supported\_Data\_Type

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、数据类型、取值范围：

```text
Operator %s does not support data type %s. The supported data type range is %s.
```

报错示例如下：

```text
Operator aclnnAdd_0_AddAiCore does not support data type DT_DUAL(25). The supported data type range is [DT_FLOAT(0), DT_FLOAT16(1), DT_INT8(2), DT_INT32(3), DT_UINT8(4), DT_INT16(6), DT_UINT16(7), DT_UINT32(8), DT_INT64(9), DT_UINT64(10), DT_DOUBLE(11), DT_BOOL(12), DT_COMPLEX64(16), DT_COMPLEX128(17), DT_BF16(27), DT_HIFLOAT8(34), DT_FLOAT8_E5M2(35), DT_FLOAT8_E4M3FN(36), DT_FLOAT8_E8M0(37), DT_FLOAT6_E3M2(38), DT_FLOAT6_E2M3(39), DT_FLOAT4_E2M1(40), DT_FLOAT4_E1M2(41)].
```

## 解决方法

需按照报错提示修改数据类型。
