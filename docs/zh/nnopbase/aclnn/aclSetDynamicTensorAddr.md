# aclSetDynamicTensorAddr<a name="ZH-CN_TOPIC_0000001983530476"></a>

## 功能说明

通过[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)使能aclOpExecutor可复用后，若**输入或输出Device内存地址变更**，需要刷新**对应aclTensorList**中记录的Device内存地址。

## 函数原型<a name="section13230182415108"></a>

```
aclnnStatus aclSetDynamicTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr)
```

## 参数说明

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="28.99%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.57%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p5336125854112"><a name="p5336125854112"></a><a name="p5336125854112"></a>executor</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p1333575820413"><a name="p1333575820413"></a><a name="p1333575820413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p1923551494413"><a name="p1923551494413"></a><a name="p1923551494413"></a>设置为复用状态的aclOpExecutor。</p>
</td>
</tr>
<tr id="row18585826161918"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p5585162612191"><a name="p5585162612191"></a><a name="p5585162612191"></a>irIndex</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p11585626201912"><a name="p11585626201912"></a><a name="p11585626201912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p205854264190"><a name="p205854264190"></a><a name="p205854264190"></a>待刷新的aclTensorList在算子原型定义中的索引，从0开始计数。</p>
</td>
</tr>
<tr id="row1277516313261"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1877519312268"><a name="p1877519312268"></a><a name="p1877519312268"></a>relativeIndex</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p127751231132618"><a name="p127751231132618"></a><a name="p127751231132618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p51918384711"><a name="p51918384711"></a><a name="p51918384711"></a>待刷新的aclTensor在aclTensorList中的索引。如果aclTensorList有N个Tensor，其取值范围为[0, N-1]。</p>
</td>
</tr>
<tr id="row12946153011917"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1946163071915"><a name="p1946163071915"></a><a name="p1946163071915"></a>tensors</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p1294653041912"><a name="p1294653041912"></a><a name="p1294653041912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p3946193010191"><a name="p3946193010191"></a><a name="p3946193010191"></a>待刷新的aclTensorList指针。</p>
</td>
</tr>
<tr id="row22097340193"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1020918345198"><a name="p1020918345198"></a><a name="p1020918345198"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p1820993461913"><a name="p1820993461913"></a><a name="p1820993461913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p14949182314910"><a name="p14949182314910"></a><a name="p14949182314910"></a>需要刷新到指定aclTensor中的Device存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](公共接口返回码.md)。

可能失败的原因：

-   返回561103：executor或tensors是空指针。
-   返回161002：relativeIndex\>=tensors里tensor的个数。
-   返回161002：irIndex\>=算子原型输入/输出参数的个数。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```
// 创建输入和输出的aclTensor和aclTensorList
std::vector<int64_t> shape = {1, 2, 3};
aclTensor tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor3 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor4 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor5 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list1[] = {tensor1, tensor2};
auto tensorList = aclCreateTensorList(list1, 2);
aclTensor *list2[] = {tensor4, tensor5};
auto output = aclCreateTensorList(list2, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// AddCustom算子有两个输入（aclTensorList和aclTensor），一个输出（aclTensorList）
// 调用第1段接口
aclnnAddCustomGetWorkspaceSize(tensorList, tensor3, output, &workspaceSize, &executor);
// 设置executor为可复用
aclSetAclOpExecutorRepeatable(executor);  
void *addr;
aclSetDynamicTensorAddr(executor, 0, 0, tensorList, addr); // 刷新输入tensorlist中第1个aclTensor的device地址
aclSetDynamicTensorAddr(executor, 0, 1, tensorList, addr); // 刷新输入tensorlist中第2个aclTensor的device地址
aclSetDynamicTensorAddr(executor, 0, 0, output, addr); // 刷新输出output中第1个aclTensor的device地址
aclSetDynamicTensorAddr(executor, 0, 1, output, addr); // 刷新输出output中第2个aclTensor的device地址
...
// 调用第2段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);
// 清理executor
aclDestroyAclOpExecutor(executor);  
```

