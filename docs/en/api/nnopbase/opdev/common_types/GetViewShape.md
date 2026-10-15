# GetViewShape

## Function

Obtains `ViewShape` of an `aclTensor`. ViewShape indicates the logical shape of an `aclTensor`.

Assume that `StorageShape` and `ViewShape` of an `aclTensor` are as follows:

- If `StorageShape` is \[10, 20\], the `aclTensor` is arranged in memory with dimensions \[10, 20\].
- If `ViewShape` is \[2, 5, 20\], during operator execution, the `aclTensor` can be treated as data arranged in the shape \[2, 5, 20\].

## Prototype

```cpp
const op::Shape& GetViewShape() const
```

## Parameters

None

## Returns

A `const op::Shape&` referring to shape information owned by the tensor, for example, a three-dimensional shape [10, 20, 30]. The returned reference does not extend the tensor lifetime. Use a value copy, as in the example, when an independent shape is required.

> **Note**: For details about `gert::Shape`, see "gert Namespace > Shape" in [Basic Data Structure and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
void Func(const aclTensor *input) {
    auto shape = input->GetViewShape();
}
```
