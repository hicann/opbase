# aclSetRawTensorAddr<a name="ZH-CN_TOPIC_0000002220416849"></a>

## 功能说明<a name="section36583473819"></a>

刷新aclTensor中原始记录的Device内存地址，aclTensor由[aclCreateTensor](aclCreateTensor.md)接口创建。

通常情况下，若网络需要频繁复用aclTensor（即保持shape、format等属性一致），可使用本接口刷新aclTensor原始的Device内存地址，达到复用的目的。

## 函数原型<a name="section13230182415108"></a>

```
aclnnStatus aclSetRawTensorAddr(aclTensor *tensor, void *addr)
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
<tbody><tr id="row12946153011917"><td class="cellrowborder" valign="top" width="28.99%" headers="mcps1.1.4.1.1 "><p id="p1946163071915"><a name="p1946163071915"></a><a name="p1946163071915"></a>tensor</p>
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

-   返回161001：参数tensor为空指针。

## 约束说明<a name="section141811212135015"></a>

-   必须在一阶段接口aclxxXxxGetWorkspaceSize之前或者二阶段接口aclxxXxx之后使用，不支持在一阶段与二阶段接口之间使用。
-   本接口可与[aclGetRawTensorAddr](aclGetRawTensorAddr.md)接口配套使用，查看刷新后的结果是否符合预期。

## 调用示例<a name="section14337653115116"></a>

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```
// 创建输入和输出张量inputTensor和outputTensor
std::vector<int64_t> shape = {1, 2, 3};
aclTensor inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
// 调用Xxx算子一、二阶段接口
auto ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...

void *addr1;
void *addr2;
... //申请device内存addr1， addr2
ret = aclSetRawTensorAddr(inputTensor, addr1); // 刷新输入张量inputTensor的device地址
ret = aclSetRawTensorAddr(outputTensor, addr2); // 刷新输出张量outputTensor的device地址
...
// 复用inputTensor，outputTensor后，调用Yyy算子aclnn一、二阶段接口
auto ret = aclnnYyyGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclnnYyy(workspace, workspaceSize, executor, stream);
...
```

