# aclDumpOpTensors<a name="ZH-CN_TOPIC_0000001983530440"></a>

## 功能说明<a name="section36583473819"></a>

模型执行过程中支持Dump算子输入/输出Tensor数据，方便算子输入/输出异常数据的问题定位和分析。

## 函数原型<a name="section13230182415108"></a>

```
aclnnStatus aclDumpOpTensors(const char *opType, const char *opName, aclTensor **tensors, size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream)
```

## 参数说明<a name="section75395119104"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="23.82%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="23.59%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="52.59%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>字符串，表示算子类型，例如“Add”。</p>
</td>
</tr>
</tbody>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>opName</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>字符串，表示算子名称，例如“add_custom”。</p>
</td>
</tr>
</tbody>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>tensors</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>一维张量，表示待Dump的输入/输出Tensor对象指针。注意Tensor顺序，输入Tensor在前，输出Tensor在后。</p>
</td>
</tr>
</tbody>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>inputTensorNum</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>表示待Dump的输入Tensor个数。</p>
</td>
</tr>
</tbody>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>outputTensorNum</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>表示待Dump的输出Tensor个数。</p>
</td>
</tr>
</tbody>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="23.82%" headers="mcps1.1.4.1.1 "><p id="p14938343556"><a name="p14938343556"></a><a name="p14938343556"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="23.59%" headers="mcps1.1.4.1.2 "><p id="p13937204313518"><a name="p13937204313518"></a><a name="p13937204313518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.59%" headers="mcps1.1.4.1.3 "><p id="p6936124317513"><a name="p6936124317513"></a><a name="p6936124317513"></a>指定执行任务的Stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](公共接口返回码.md)。

## 约束说明<a name="section141811212135015"></a>

本接口需要在开启算子Dump功能时有效，您可以通过aclInit接口开启Dump，也可以通过aclmdlInitDump、aclmdlSetDump、aclmdlFinalizeDump系列接口开启Dump，接口介绍请参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)。

## 调用示例<a name="section1655912368315"></a>

接口调用请参考[aclCreateTensor](aclCreateTensor.md)的调用示例。
关键代码示例如下（仅供参考，不支持直接拷贝运行）。
1. 通过aclInit接口开启算子Dump功能。关键代码如下：
```
// AscendCL Init
aclInit("./acl.json");
aclrtSetDevice(0);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
```
acl.json示例如下（具体参见aclInit接口文档中模型Dump配置、单算子Dump配置示例）：
```
{
    "dump": [
        "dump_path": "./",
        "dump_list": [],
        "dump_mode": "all",
        "dump_data": "tensor"
    ]
}
```
2. 本接口调用的关键伪代码（以torch算子为例）如下：
```
// 2.1 将at::Tensor转换成aclTensor对象
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

// 2.2 生成待Dump的输入/输出Tensor对象指针一维张量
#define INIT_ACL_TENSOR_ARRAY(tensors, ...) aclTensor* tensors[] = {VA_ARGS}

// 2.3 调用接口Dump输入/输出Tensor
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
