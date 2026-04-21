# aclDumpOpTensors

## 功能说明

模型执行过程中支持Dump算子输入/输出Tensor数据，方便算子输入/输出异常数据的问题定位和分析。

## 函数原型

```cpp
aclnnStatus aclDumpOpTensors(const char *opType, const char *opName, aclTensor **tensors, size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opType | 输入 | 字符串，表示算子类型，例如“Add”。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](公共接口返回码.md)。

## 约束说明

本接口需要在开启算子Dump功能时有效，您可以通过aclInit接口开启Dump，也可以通过aclmdlInitDump、aclmdlSetDump、aclmdlFinalizeDump系列接口开启Dump，接口介绍请参见《Runtime运行时 API》。

## 调用示例

关键代码示例如下（仅供参考，不支持直接拷贝运行）。

1. 通过aclInit接口开启算子Dump功能。关键代码如下：

    ```cpp
    // 资源初始化
    aclInit("./acl.json");
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ```

    acl.json示例如下（具体参见aclInit接口文档中模型Dump配置、单算子Dump配置示例）：

    ```json
    {
        "dump": [
            "dump_path": "./",
            "dump_list": [],
            "dump_mode": "all",
            "dump_data": "tensor"
        ]
    }
    ```

2. 调用本接口关键伪代码（以torch算子为例）如下：

    ```cpp
    // 1. 将at::Tensor转换成aclTensor对象
    aclTensor *ConvertTensor(const at::Tensor &at_tensor)
    {
        at::ScalarType scalar_data_type = at_tensor.scalar_type();
        aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
        c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;

        const auto dimNum = at_tensor.sizes().size();
        aclFormat format = ACL_FORMAT_ND;
        if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
            format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
            // if acl_data_type is ACL_STRING, storageDims is empty.
            if (acl_data_type != ACL_STRING) {
                storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
            }
        } else {
            switch (dimNum) {
                case NCL_DIM_NUM:
                    format = ACL_FORMAT_NCL;
                    break;
                case NCHW_DIM_NUM:
                    format = ACL_FORMAT_NCHW;
                    break;
                case NCDHW_DIM_NUM:
                    format = ACL_FORMAT_NCDHW;
                    break;
                default:
                    format = ACL_FORMAT_ND;
            }
            // if acl_data_type is ACL_STRING, storageDims is empty.
            if (acl_data_type != ACL_STRING) {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
            }
        }

        if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
            c10::Scalar expScalar = at_tensor.item();
            at::Tensor aclInput = at_npu::native::OpPreparation::copy_scalar_to_device(expScalar, scalar_data_type);
            return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                            aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                            storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;

    }

    // 2. 生成待Dump的输入/输出Tensor对象指针一维张量
    #define INIT_ACL_TENSOR_ARRAY(tensors, ...) aclTensor* tensors[] = {VA_ARGS}

    // 3. 调用接口Dump输入/输出Tensor
    namespace ascendc_ops {
    at::Tensor ascendc_add(const at::Tensor& x, const at::Tensor& y)
    {
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        uint32_t numBlocks = 8;
        uint32_t totalLength = 1;
        for (uint32_t size : x.sizes()) {
        totalLength = size;
        }
        add_custom<<<numBlocks, nullptr, aclStream>>>((uint8_t)(x.mutable_data_ptr()), (uint8_t*)(y.mutable_data_ptr()), (uint8_t*)(z.mutable_data_ptr()), totalLength);
        INIT_ACL_TENSOR_ARRAY(tensors, ConvertTensor(x), ConvertTensor(y), ConvertTensor(z));
        // Dump算子输入/输出Tensor数据
        aclDumpOpTensors("Add", "add_custom", tensors, 2, 1, aclStream);
        return z;
    }
    }
    ```
