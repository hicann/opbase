# GetOriginalShape

## Function

Obtains `OriginShape` of an `aclTensor`.

`OriginShape` is the original shape of an `aclTensor` before it passes through a `transdata` node (if any), that is, the mathematical description of the tensor's shape.

## Prototype

```cpp
const op::Shape& GetOriginalShape() const
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
    auto shape = input->GetOriginalShape();
}
```
