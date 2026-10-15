# IsUnknownShape

## 功能说明

图模式场景下，检查输入shape是否至少有一根轴的长度为不确定值（-1）。

## 函数原型

```cpp
bool IsUnknownShape(const gert::Shape &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 待判定的输入shape。 |

## 返回值说明

返回类型为bool。

- true：输入shape至少有一根轴的长度为不确定值（-1）。
- false：输入shape的所有轴长度均不是不确定值（-1）。

例如，shape为`[-1, 4]`时返回true，shape为`[3, 4]`时返回false。未知秩由[IsUnknownRank](IsUnknownRank.md)单独判断。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
if (Ops::Base::IsUnknownShape(*inputShape)) {
    output_y_shape->SetDimNum(input_dim_num);
}
```
