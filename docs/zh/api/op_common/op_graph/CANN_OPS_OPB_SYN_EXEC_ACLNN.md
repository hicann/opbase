# CANN\_OPS\_OPB\_SYN\_EXEC\_ACLNN

## 功能说明

Aclnn接口在图场景的通用执行接口（同步接口），内部串行调用两阶段执行。

## 函数原型

```cpp
CANN_OPS_OPB_SYN_EXEC_ACLNN(ctx, aclnnApi, ...)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| ctx | 输入 | 算子执行上下文（OpExecuteContext），用于分配Workspace和获取Stream。 |
| aclnnApi | 输入 | 待回调执行的ACLNN算子API名称，如aclnnScatterList。 |
| ... | 输入 | 算子输入、输出及属性参数，支持gert::Tensor、gert::Tensor列表、标量、属性等类型。 |

## 返回值说明

返回int类型，GRAPH_SUCCESS表示执行成功，GRAPH_FAILED表示执行失败。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(hostApiCtx, aclnnScatterList, geTenserListVar, indices, update, mask,
                                          reduce, *axis);
OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE(hostApiCtx->GetNodeName(), "apiRet faild:%d", apiRet),
            return GRAPH_FAILED);
```
