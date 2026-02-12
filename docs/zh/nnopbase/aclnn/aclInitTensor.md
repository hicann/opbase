# aclInitTensor<a name="ZH-CN_TOPIC_0000001983530468"></a>

## 功能说明<a name="section36583473819"></a>

初始化给定aclTensor的参数，aclTensor由[aclCreateTensor](aclCreateTensor.md)接口创建。

当用户想复用已创建aclTensor时，可使用该接口先重置aclTensor的各项属性。

## 函数原型<a name="section13230182415108"></a>

```
aclnnStatus aclInitTensor(aclTensor *tensor, const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storageDims, uint64_t storageDimsNum, void *tensorDataAddr)
```

## 参数说明<a name="section75395119104"></a>

> **说明：**
>-   关于aclTensor的StorageShape和ViewShape：
>    -   ViewShape表示Tensor的逻辑shape，是Tensor在实际使用时需要用到的大小。
>    -   StorageShape表示Tensor的实际物理排布shape，是Tensor在内存上实际存在的大小。
>    举例如下：
>    -   StorageShape为\[10, 20\]：表示该Tensor在内存上是按照\[10, 20\]排布的。
>    -   ViewShape为\[2, 5, 20\]：在算子使用时，表示该Tensor可被视为一块\[2, 5, 20\]的数据使用。

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="29.03%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.529999999999998%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p5336125854112"><a name="p5336125854112"></a><a name="p5336125854112"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p1333575820413"><a name="p1333575820413"></a><a name="p1333575820413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p1923551494413"><a name="p1923551494413"></a><a name="p1923551494413"></a>待初始化参数的aclTensor。</p>
</td>
</tr>
<tr id="row1673920194218"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1373930154210"><a name="p1373930154210"></a><a name="p1373930154210"></a>viewDims</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p117398094214"><a name="p117398094214"></a><a name="p117398094214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p1659125431910"><a name="p1659125431910"></a><a name="p1659125431910"></a>tensor的ViewShape维度值，<span>为非负整数</span>。</p>
</td>
</tr>
<tr id="row11875101874219"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p98751518174213"><a name="p98751518174213"></a><a name="p98751518174213"></a>viewDimsNum</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p1387581810421"><a name="p1387581810421"></a><a name="p1387581810421"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p85911454111918"><a name="p85911454111918"></a><a name="p85911454111918"></a>tensor的ViewShape维度数。</p>
</td>
</tr>
<tr id="row17593925422"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p45938294213"><a name="p45938294213"></a><a name="p45938294213"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p75934234213"><a name="p75934234213"></a><a name="p75934234213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p7591175414195"><a name="p7591175414195"></a><a name="p7591175414195"></a>tensor的数据类型。</p>
</td>
</tr>
<tr id="row1047272212427"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p17472922124220"><a name="p17472922124220"></a><a name="p17472922124220"></a>stride</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p34721922184219"><a name="p34721922184219"></a><a name="p34721922184219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p559175441915"><a name="p559175441915"></a><a name="p559175441915"></a>tensor各维度元素的访问步长<span>，为非负整数</span>。</p>
</td>
</tr>
<tr id="row142925372239"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p333310582415"><a name="p333310582415"></a><a name="p333310582415"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p8333195834115"><a name="p8333195834115"></a><a name="p8333195834115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p1359175415193"><a name="p1359175415193"></a><a name="p1359175415193"></a>tensor首元素相对于storage的偏移，<span>为非负整数</span>。</p>
</td>
</tr>
<tr id="row889032054219"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1689022016422"><a name="p1689022016422"></a><a name="p1689022016422"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p1089032064210"><a name="p1089032064210"></a><a name="p1089032064210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p18591115441916"><a name="p18591115441916"></a><a name="p18591115441916"></a>tensor的数据排布格式。</p>
</td>
</tr>
<tr id="row1241121724218"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1424131764218"><a name="p1424131764218"></a><a name="p1424131764218"></a>storageDims</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p124113171427"><a name="p124113171427"></a><a name="p124113171427"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p18592554151917"><a name="p18592554151917"></a><a name="p18592554151917"></a>tensor的StorageShape维度值，<span>为非负整数</span>。</p>
</td>
</tr>
<tr id="row656931364214"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1656981316427"><a name="p1656981316427"></a><a name="p1656981316427"></a>storageDimsNum</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p756951374213"><a name="p756951374213"></a><a name="p756951374213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p559245471916"><a name="p559245471916"></a><a name="p559245471916"></a>tensor的StorageShape维度数。</p>
</td>
</tr>
<tr id="row3577101515428"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1578161515425"><a name="p1578161515425"></a><a name="p1578161515425"></a>tensorDataAddr</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p165782015134216"><a name="p165782015134216"></a><a name="p165782015134216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p6592154141914"><a name="p6592154141914"></a><a name="p6592154141914"></a>tensor在Device侧的存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](公共接口返回码.md)。

## 约束说明<a name="section141811212135015"></a>

无

## 调用示例<a name="section423216448015"></a>

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```
std::vector<int64_t> viewDims = {2, 4};
std::vector<int64_t> stride = {4, 1};
std::vector<int64_t> storageDims = {2, 4};  
// tensor为复用的已经创建的aclTensor
// deviceAddr为tensor在Device侧的存储地址
auto ret = aclInitTensor(tensor, viewDims.data(), viewDims.size(), ACL_FLOAT16, stride.data(), 0, aclFormat::ACL_FORMAT_ND, storageDims.data(), storageDims.size(), deviceAddr);
```

