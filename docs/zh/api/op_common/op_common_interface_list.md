# op\_common接口列表

介绍CANN算子调用或开发时依赖的公共日志、能力，主要涉及如下几类。

- log：算子日志打印相关宏。
- math：数学计算相关接口。
- platform：算子执行平台信息相关接口。
- infershape：算子shape处理相关接口。

**说明：表1中接口为正式接口，支持版本向后兼容；不在表中的接口为预留接口，用户暂不需要关注。**

**表 1**  op\_common接口列表

<table><thead>
  <tr>
    <th>分类</th>
    <th>接口</th>
    <th>说明</th>
    <th>所属头文件</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="33">log</td>
    <td><a href='OP_LOGE.md'>OP_LOGE</a></td>
    <td>打印算子ERROR级别日志。</td>
    <td rowspan="33">include/op_common/log/log.h</td>
  </tr>
  <tr>
    <td><a href='OP_LOGD.md'>OP_LOGD</a></td>
    <td>打印算子DEBUG级别日志。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGI.md'>OP_LOGI</a></td>
    <td>打印算子INFO级别日志。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGW.md'>OP_LOGW</a></td>
    <td>打印算子WARNING级别日志。</td>
  </tr>
  <tr>
    <td><a href='OP_CHECK_IF.md'>OP_CHECK_IF</a></td>
    <td>当condition条件成立时，输出日志，并执行return表达式。</td>
  </tr>
  <tr>
    <td><a href='OP_CHECK_NULL_WITH_CONTEXT.md'>OP_CHECK_NULL_WITH_CONTEXT</a></td>
    <td>根据传入的context上下文，校验传入的指针是否为nullptr。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_INPUT_SHAPE.md'>OP_LOGE_WITH_INVALID_INPUT_SHAPE</a></td>
    <td>记录并上报算子输入形状校验错误，上报EZ0001错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_ATTR.md'>OP_LOGE_WITH_INVALID_ATTR</a></td>
    <td>记录并上报算子属性值校验错误，上报EZ0002错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_ATTR_SIZE.md'>OP_LOGE_WITH_INVALID_ATTR_SIZE</a></td>
    <td>记录并上报算子属性大小校验错误，上报EZ0003错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_INPUT.md'>OP_LOGE_WITH_INVALID_INPUT</a></td>
    <td>记录并上报算子输入参数为空校验错误，上报EZ0004错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE.md'>OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE</a></td>
    <td>记录并上报算子输入形状大小校验错误，上报EZ0005错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_INPUT_FORMAT.md'>OP_LOGE_WITH_INVALID_INPUT_FORMAT</a></td>
    <td>记录并上报算子输入格式校验错误，上报EZ0006错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_WITH_INVALID_INPUT_DTYPE.md'>OP_LOGE_WITH_INVALID_INPUT_DTYPE</a></td>
    <td>记录并上报算子输入数据类型校验错误，上报EZ0007错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPE.md'>OP_LOGE_FOR_INVALID_SHAPE</a></td>
    <td>记录并上报参数形状校验错误，上报EZ0008错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON</a></td>
    <td>记录并上报参数形状校验错误（带原因），上报EZ0009错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON</a></td>
    <td>记录并上报多参数形状校验错误（带原因），上报EZ0010错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPEDIM.md'>OP_LOGE_FOR_INVALID_SHAPEDIM</a></td>
    <td>记录并上报参数形状维度校验错误，上报EZ0011错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON</a></td>
    <td>记录并上报参数形状维度校验错误（带原因），上报EZ0012错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON</a></td>
    <td>记录并上报多参数形状维度校验错误（带原因），上报EZ0013错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPESIZE.md'>OP_LOGE_FOR_INVALID_SHAPESIZE</a></td>
    <td>记录并上报参数形状大小校验错误，上报EZ0014错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON</a></td>
    <td>记录并上报参数形状大小校验错误（带原因），上报EZ0015错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON</a></td>
    <td>记录并上报多参数形状大小校验错误（带原因），上报EZ0016错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_FORMAT.md'>OP_LOGE_FOR_INVALID_FORMAT</a></td>
    <td>记录并上报参数格式校验错误，上报EZ0017错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON</a></td>
    <td>记录并上报多参数格式校验错误（带原因），上报EZ0018错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_DTYPE.md'>OP_LOGE_FOR_INVALID_DTYPE</a></td>
    <td>记录并上报参数数据类型校验错误，上报EZ0019错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON</a></td>
    <td>记录并上报参数数据类型校验错误（带原因），上报EZ0020错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON</a></td>
    <td>记录并上报多参数数据类型校验错误（带原因），上报EZ0021错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_TENSORNUM.md'>OP_LOGE_FOR_INVALID_TENSORNUM</a></td>
    <td>记录并上报参数张量数量校验错误，上报EZ0022错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON</a></td>
    <td>记录并上报多参数张量数量校验错误（带原因），上报EZ0023错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_VALUE.md'>OP_LOGE_FOR_INVALID_VALUE</a></td>
    <td>记录并上报参数值校验错误，上报EZ0024错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_LISTSIZE.md'>OP_LOGE_FOR_INVALID_LISTSIZE</a></td>
    <td>记录并上报参数列表大小校验错误，上报EZ0025错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_VALUE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_VALUE_WITH_REASON</a></td>
    <td>记录并上报参数值校验错误（带原因），上报EZ0026错误码。</td>
  </tr>
  <tr>
    <td><a href='OP_LOGE_FOR_INVALID_VALUES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_VALUES_WITH_REASON</a></td>
    <td>记录并上报多参数值校验错误（带原因），上报EZ0027错误码。</td>
  </tr>
  <tr>
    <td rowspan="5">math</td>
    <td><a href='FloorDiv.md'>FloorDiv</a></td>
    <td>向下取整的除法。</td>
    <td rowspan="5">include/op_common/op_host/util/math_util.h</td>
  </tr>
  <tr>
    <td><a href='FloorAlign.md'>FloorAlign</a></td>    
    <td>以align为单元，向下对齐。</td>
  </tr>
  <tr>
    <td><a href='CeilDiv.md'>CeilDiv</a></td>    
    <td>向上取整的除法。</td>
  </tr>
  <tr>
    <td><a href='CeilAlign.md'>CeilAlign</a></td>    
    <td>以align为单元，向上对齐。</td>
  </tr>
  <tr>
    <td><a href='IsFloatEqual.md'>IsFloatEqual</a></td>    
    <td>判断两个float类型或double类型的数值是否相等。</td>
  </tr>
  <tr>
    <td rowspan="4">platform</td>
    <td><a href='GetAivCoreNum.md'>GetAivCoreNum</a></td>
    <td>获取平台AI Vector的核数。</td>
    <td rowspan="4">include/op_common/op_host/util/platform_util.h</td>
  </tr>
  <tr>
    <td><a href='GetAicCoreNum.md'>GetAicCoreNum</a></td>    
    <td>获取平台AI Cube的核数。</td>
  </tr>
  <tr>
    <td><a href='GetUbSize.md'>GetUbSize</a></td>    
    <td>获取平台UB（Unified Buffer）空间大小，单位bytes。</td>
  </tr>
  <tr>
    <td><a href='GetUbBlockSize.md'>GetUbBlockSize</a></td>    
    <td>获取平台UB（Unified Buffer）的block单元大小，单位bytes。</td>
  </tr>
  <tr>
    <td rowspan="6">infershape</td>
    <td><a href='SetUnknownRank.md'>SetUnknownRank</a></td>
    <td>设置输入shape为维度不确定的动态shape。</td>
    <td rowspan="4">include/op_common/op_host/util/shape_util.h</td>
  </tr>
  <tr>
    <td><a href='IsUnknownRank.md'>IsUnknownRank</a></td>    
    <td>检查输入shape是否为维度不确定的shape。</td>
  </tr>
  <tr>
    <td><a href='SetUnknownShape.md'>SetUnknownShape</a></td>    
    <td>设置输入shape的维度为rank，且每根轴长度都为不确定值。</td>
  </tr>
  <tr>
    <td><a href='IsUnknownShape.md'>IsUnknownShape</a></td>    
    <td>检查输入shape的每一根轴长度是否都为不确定值。</td>
  </tr>
  <tr>
    <td><a href='InferShape4Broadcast.md'>InferShape4Broadcast</a></td>    
    <td>broadcast类算子的infershape方法。</td>
    <td rowspan="1">include/op_common/op_host/infershape_broadcast_util.h</td>
  </tr>
  <tr>
    <td><a href='InferShape4Elewise.md'>InferShape4Elewise</a></td>    
    <td>elewise类算子的infershape方法。</td>
    <td rowspan="1">include/op_common/op_host/infershape_elewise_util.h</td>
  </tr>
</tbody>
</table> 
