# GetViewStrides

## Function

Obtains `ViewStrides` of an `aclTensor`.

## Prototype

```cpp
const op::Strides& GetViewStrides() const
```

## Parameters

None

## Returns

A `const op::Strides&` referring to the tensor's view strides. The returned reference does not extend the tensor lifetime. The example makes an independent value copy.

## Restrictions

None

## Examples

```cpp
// Obtain the view stride of the input and print the stride of each dimension in sequence.
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    for (int64_t stride : strides) {
        std::cout << stride << std::endl;
    }
}
```
