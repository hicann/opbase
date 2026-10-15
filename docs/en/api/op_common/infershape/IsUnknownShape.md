# IsUnknownShape

## Function

Checks whether at least one input-shape dimension has an unknown length (-1) in graph mode.

## Prototype

```cpp
bool IsUnknownShape(const gert::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| Input shape to be checked.|

## Returns

Return type: `bool`

- **true**: At least one input-shape dimension has an unknown length (-1).
- **false**: No input-shape dimension has an unknown length (-1).

For example, `[-1, 4]` returns `true`, while `[3, 4]` returns `false`.
Unknown rank is checked separately by [IsUnknownRank](IsUnknownRank.md).

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
if (Ops::Base::IsUnknownShape(*inputShape)) {
    output_y_shape->SetDimNum(input_dim_num);
}
```
