# GetStorageShape

## Function

Obtains `StorageShape` of an `aclTensor`.

StorageShape indicates the actual runtime shape of an `aclTensor` in memory. It reflects how `OriginShape` is rearranged under the selected `StorageFormat`.

## Prototype

```cpp
const op::Shape& GetStorageShape() const
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
    auto shape = input->GetStorageShape();
}
```
