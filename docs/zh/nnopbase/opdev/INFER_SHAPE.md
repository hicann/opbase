# INFER\_SHAPE<a name="ZH-CN_TOPIC_0000002515682991"></a>

## 宏功能<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_section0397626155918"></a>

针对指定算子，运行其InferShape函数，推导输出shape。

## 宏原型<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_section1373361681114"></a>

```
INFER_SHAPE(KERNEL_NAME, op_args...)
```

## 参数说明<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_section76104422125"></a>

<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_table4634135492614"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_row1963435442618"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p14634454132618"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p14634454132618"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p14634454132618"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.409999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p4634135452615"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p4634135452615"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p4634135452615"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.36999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p763465418263"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p763465418263"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p763465418263"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_row8634185442615"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p193617166232"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p193617166232"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p193617166232"></a>KERNEL_NAME</p>
</td>
<td class="cellrowborder" valign="top" width="15.409999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1634195412618"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1634195412618"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1634195412618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.36999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p19964198101710"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p19964198101710"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p19964198101710"></a>算子名，例如Add。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_row594913243139"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1695072418137"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1695072418137"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1695072418137"></a>op_args...</p>
</td>
<td class="cellrowborder" valign="top" width="15.409999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p149507245135"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p149507245135"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p149507245135"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.36999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1295062411134"><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1295062411134"></a><a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_p1295062411134"></a>算子的参数，包括输入<a href="OP_INPUT.md">OP_INPUT</a>、输出<a href="OP_OUTPUT.md">OP_OUTPUT</a>、属性<a href="OP_ATTR.md">OP_ATTR</a>等参数。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_zh-cn_topic_0000001647821269_section19165124931511"></a>

-   如果算子需要INFER\_SHAPE，那么此宏需要在[ADD\_TO\_LAUNCHER\_LIST\_AICORE](ADD_TO_LAUNCHER_LIST_AICORE.md)之前调用。
-   如下接口是上述宏定义会调用到的关联接口。

    ```
    OP_INPUT(x...)
    OP_OUTPUT(x...)
    OP_ATTR(x...)
    OP_WORKSPACE(x...)
    OP_OUTSHAPE(x...)
    OP_OPTION(x...)
    OP_EMPTY_ARG
    OP_MODE(x...)
    ```

## 调用示例<a name="zh-cn_topic_0000002114052997_zh-cn_topic_0000001944142092_section1086173012323"></a>

```
// 调用INFER_SHAPE推导batchmatmul算子的输出shape，其中BatchMatMulV3是算子的名字，OP_INPUT是算子输入参数，OP_OUTPUT是算子输出参数，OP_ATTR是算子的属性参数
INFER_SHAPE(BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum));
```

