# aclSetTensorAddr<a name="ZH-CN_TOPIC_0000001983690200"></a>

## 功能说明<a name="section36583473819"></a>

通过[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)使能aclOpExecutor可复用后，若**输入或输出Device内存地址变更**，需要刷新**对应aclTensor**中记录的Device内存地址。

## 函数原型<a name="section13230182415108"></a>

```
aclnnStatus aclSetTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr)
```

## 参数说明<a name="section75395119104"></a>

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
<tr id="row18585826161918"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p5585162612191"><a name="p5585162612191"></a><a name="p5585162612191"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p11585626201912"><a name="p11585626201912"></a><a name="p11585626201912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p205854264190"><a name="p205854264190"></a><a name="p205854264190"></a>待刷新的aclTensor索引，取值范围是[0, tensor的总数-1]。</p>
</td>
</tr>
<tr id="row12946153011917"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1946163071915"><a name="p1946163071915"></a><a name="p1946163071915"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p1294653041912"><a name="p1294653041912"></a><a name="p1294653041912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p3946193010191"><a name="p3946193010191"></a><a name="p3946193010191"></a>待刷新的aclTensor指针。</p>
</td>
</tr>
<tr id="row22097340193"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1020918345198"><a name="p1020918345198"></a><a name="p1020918345198"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="24.57%" headers="mcps1.1.4.1.2 "><p id="p1820993461913"><a name="p1820993461913"></a><a name="p1820993461913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p72134196915"><a name="p72134196915"></a><a name="p72134196915"></a>需要刷新到指定aclTensor中的Device存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](公共接口返回码.md)。

可能失败的原因：

-   返回561103：executor或tensor是空指针。
-   返回161002：index取值越界。
-   返回161002：第一次执行一阶段接口aclxxXxxGetWorkspaceSize时传入的aclTensor是nullptr，不再支持刷新地址。

## 约束说明<a name="section141811212135015"></a>

无

## 调用示例<a name="section14337653115116"></a>

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
aclTensor output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor1, tensor2};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// AddCustom算子有两个输入（aclTensorList和aclTensor），一个输出（aclTensor）
// 调用第1段接口
aclnnAddCustomGetWorkspaceSize(tensorList, tensor3, output, &workspaceSize, &executor);
// 设置executor为可复用
aclSetAclOpExecutorRepeatable(executor);  
void *addr;
aclSetTensorAddr(executor, 0, tensor1, addr); // 刷新输入tensorlist中第1个aclTensor的device地址
aclSetTensorAddr(executor, 1, tensor2, addr); // 刷新输入tensorlist中第2个aclTensor的device地址
aclSetTensorAddr(executor, 2, tensor3, addr); // 刷新输入aclTensor的device地址
aclSetTensorAddr(executor, 3, output, addr); // 刷新输出aclTensor的device地址
...
// 调用第2段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);
// 清理executor
aclDestroyAclOpExecutor(executor);  
```

