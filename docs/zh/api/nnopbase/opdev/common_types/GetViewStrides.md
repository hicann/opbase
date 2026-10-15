# GetViewStrides

## 功能说明

获取aclTensor的ViewStrides。

## 函数原型

```cpp
const op::Strides& GetViewStrides() const
```

## 参数说明

无

## 返回值说明

返回const op::Strides&，引用tensor持有的view strides。返回引用不会延长tensor的生命周期；示例按值复制得到独立对象。

## 约束说明

无

## 调用示例

```cpp
// 获取input的View Stride并依次打印出每一维的stride大小
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    for (int64_t stride : strides) {
        std::cout << stride << std::endl;
    }
}
```
