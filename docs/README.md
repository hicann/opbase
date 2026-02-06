# 项目文档
## 基本概念

- <a name="zh-cn_topic_0000002114052989_zh-cn_topic_0000001917744184_li1543255153717"></a>**Level0层接口**

  简称L0接口，表示调用单Kernel的Host侧API，提供了细颗粒API（单Kernel下发）和算子API开发的基础结构体（如Tensor定义等）和公共基础能力（如workspace复用、引擎调度等），上层应用或者L2接口可通过L0接口的快速组装实现高性能计算。

  L0接口返回值类型是Tensor的类型结构，如aclTensor\*、std::tuple\<aclTensor\*, aclTensor\*\>、aclTensorList\*，最后一个参数固定为aclOpExecutor \*executor，类型与名称均不可变，示例如下：

  ```Cpp
  aclTensor* AddNd(aclTensor *x1, aclTensor *x2, aclOpExecutor *executor)
  ```

  L0接口命名空间为“namespace l0op”，接口名为“\$\{op\_type\}\$\{format\}\$\{dtype\}”，其中\$\{op\_type\}为算子名，\$\{format\}为算子输入/输出数据格式，\$\{dtype\}为算子输入/输出数据类型（对于非常规的输入/输出数据类型，需带上数据类型匹配关系）。调用示例如下：

  ```Cpp
  l0op::AddNd                          //Add算子输入均按ND计算
  l0op::MatMulNdFp162Fp32              //MatMul算子输入输出均按ND格式计算，并且2代表“To”，表示输入fp16、输出fp32
  l0op::MatMulNzFp162Fp16              //MatMul算子输入输出均按NZ格式计算，并且2代表“To”，表示输入与输出都是fp16
  ```

- <a name="zh-cn_topic_0000002114052989_zh-cn_topic_0000001917744184_li572474119382"></a>**Level2层接口**

  简称L2接口，是对L0接口的高层级封装（内部通过调用单个或多个L0接口实现更灵活功能），表示更上层的Host侧API（**aclnn API通常被称为L2接口**）。该类接口提供单算子直调方式，屏蔽了算子内部实现逻辑，用户直接调用L2接口即可实现调用算子。

  L2接口返回值类型是aclnnStatus，一般包括获取workspaceSize和算子执行“两段式接口”：

  ```Cpp
  aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
  aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
  ```

  -   aclnnXxxGetWorkspaceSize最后两个参数固定为\(uint64\_t \*workspaceSize, aclOpExecutor \*\*executor\)，名称和类型均不可变。
  -   aclnnXxx接口参数固定为\(void \*workspace, uint64\_t workspaceSize, aclOpExecutor \*executor, aclrtStream stream\)。

  其中aclnnXxxGetWorkspaceSize为第一段接口，主要用于计算本次API调用过程中需要多少workspace内存，获取到本次计算所需的workspaceSize后，按照workspaceSize申请NPU内存，然后调用第二段接口aclnnXxx执行计算。“Xxx”表示对应的算子类型，如Add算子。

  > **说明：** 
  >
  >- workspace是指除输入/输出外，API在AI处理器上完成计算所需要的临时内存。
  >
  >- 二阶段接口aclnnXxx\(...\)不能重复调用，如下调用方式会出现异常：
  >
  >  ```
  >  aclnnXxxGetWorkspaceSize(...)
  >  aclnnXxx(...)
  >  aclnnXxx(...)
  >  ```

- **原地算子接口**

  表示在原地址进行更新操作的算子接口，其计算过程中输入和输出为同一地址，以减少不必要的内存占用。aclnn类原地算子接口名一般定义为：aclnnInplaceXxxGetWorkspaceSize（一阶段接口）、aclnnInplaceXxx（二阶段接口）。

## nnopbase接口

本章为aclnn API调用提供基础操作能力，如Tensor创建/释放、Tensor信息获取等，称为**aclnn公共接口**。

此外，还提供Ascend C自动生成aclnn API和内置算子aclnn API开发场景依赖的框架调度能力，如aclnn缓存、workspace复用等，称为**aclnn开发接口**。

### aclnn
aclnn API调用时依赖的公共Meta接口，如创建/释放aclTensor、aclScalar、aclIntArray等。此类接口对应本项目`include/nnopbase/aclnn`下头文件的定义。

-   [aclnn公共接口列表](zh/nnopbase/aclnn/公共接口列表.md)
-   [aclCreateBoolArray](zh/nnopbase/aclnn/aclCreateBoolArray.md)
-   [aclCreateFloatArray](zh/nnopbase/aclnn/aclCreateFloatArray.md)
-   [aclCreateIntArray](zh/nnopbase/aclnn/aclCreateIntArray.md)
-   [aclCreateScalar](zh/nnopbase/aclnn/aclCreateScalar.md)
-   [aclCreateScalarList](zh/nnopbase/aclnn/aclCreateScalarList.md)
-   [aclCreateTensor](zh/nnopbase/aclnn/aclCreateTensor.md)
-   [aclCreateTensorList](zh/nnopbase/aclnn/aclCreateTensorList.md)
-   [aclDestroyAclOpExecutor](zh/nnopbase/aclnn/aclDestroyAclOpExecutor.md)
-   [aclDestroyBoolArray](zh/nnopbase/aclnn/aclDestroyBoolArray.md)
-   [aclDestroyFloatArray](zh/nnopbase/aclnn/aclDestroyFloatArray.md)
-   [aclDestroyIntArray](zh/nnopbase/aclnn/aclDestroyIntArray.md)
-   [aclDestroyScalar](zh/nnopbase/aclnn/aclDestroyScalar.md)
-   [aclDestroyScalarList](zh/nnopbase/aclnn/aclDestroyScalarList.md)
-   [aclDestroyTensor](zh/nnopbase/aclnn/aclDestroyTensor.md)
-   [aclDestroyTensorList](zh/nnopbase/aclnn/aclDestroyTensorList.md)
-   [aclDumpOpTensors](zh/nnopbase/aclnn/aclDumpOpTensors.md)
-   [aclGetBoolArraySize](zh/nnopbase/aclnn/aclGetBoolArraySize.md)
-   [aclGetDataType](zh/nnopbase/aclnn/aclGetDataType.md)
-   [aclGetFloatArraySize](zh/nnopbase/aclnn/aclGetFloatArraySize.md)
-   [aclGetFormat](zh/nnopbase/aclnn/aclGetFormat.md)
-   [aclGetIntArraySize](zh/nnopbase/aclnn/aclGetIntArraySize.md)
-   [aclGetRawTensorAddr](zh/nnopbase/aclnn/aclGetRawTensorAddr.md)
-   [aclGetScalarListSize](zh/nnopbase/aclnn/aclGetScalarListSize.md)
-   [aclGetStorageShape](zh/nnopbase/aclnn/aclGetStorageShape.md)
-   [aclGetTensorListSize](zh/nnopbase/aclnn/aclGetTensorListSize.md)
-   [aclGetViewOffset](zh/nnopbase/aclnn/aclGetViewOffset.md)
-   [aclGetViewShape](zh/nnopbase/aclnn/aclGetViewShape.md)
-   [aclGetViewStrides](zh/nnopbase/aclnn/aclGetViewStrides.md)
-   [aclInitTensor](zh/nnopbase/aclnn/aclInitTensor.md)
-   [aclSetAclOpExecutorRepeatable](zh/nnopbase/aclnn/aclSetAclOpExecutorRepeatable.md)
-   [aclSetDynamicInputTensorAddr](zh/nnopbase/aclnn/aclSetDynamicInputTensorAddr.md)
-   [aclSetDynamicOutputTensorAddr](zh/nnopbase/aclnn/aclSetDynamicOutputTensorAddr.md)
-   [aclSetDynamicTensorAddr](zh/nnopbase/aclnn/aclSetDynamicTensorAddr.md)
-   [aclSetInputTensorAddr](zh/nnopbase/aclnn/aclSetInputTensorAddr.md)
-   [aclSetOutputTensorAddr](zh/nnopbase/aclnn/aclSetOutputTensorAddr.md)
-   [aclSetRawTensorAddr](zh/nnopbase/aclnn/aclSetRawTensorAddr.md)
-   [aclSetTensorAddr](zh/nnopbase/aclnn/aclSetTensorAddr.md)
-   [aclnnInit](zh/nnopbase/aclnn/aclnnInit.md)
-   [aclnnFinalize](zh/nnopbase/aclnn/aclnnFinalize.md)
-   [预留接口](zh/nnopbase/aclnn/预留接口.md)
-   [公共接口返回码](zh/nnopbase/aclnn/公共接口返回码.md)

### opdev

aclnn API开发时依赖的框架基础能力接口，如算子执行器（opExecutor）处理、数据类型/格式/shape等。此类接口对应本项目`include/nnopbase/opdev`下头文件的定义。

-   [aclnn开发接口列表](zh/nnopbase/opdev/aclnn开发接口列表.md)
-   [常用宏和类](zh/nnopbase/opdev/常用宏和类.md)
    -   [ADD\_TO\_LAUNCHER\_LIST\_AICORE](zh/nnopbase/opdev/ADD_TO_LAUNCHER_LIST_AICORE.md)
    -   [ADD\_TO\_LAUNCHER\_LIST\_AICPU](zh/nnopbase/opdev/ADD_TO_LAUNCHER_LIST_AICPU.md)
    -   [CREATE\_EXECUTOR](zh/nnopbase/opdev/CREATE_EXECUTOR.md)
    -   [DFX\_IN](zh/nnopbase/opdev/DFX_IN.md)
    -   [DFX\_OUT](zh/nnopbase/opdev/DFX_OUT.md)
    -   [INFER\_SHAPE](zh/nnopbase/opdev/INFER_SHAPE.md)
    -   [L0\_DFX](zh/nnopbase/opdev/L0_DFX.md)
    -   [L2\_DFX\_PHASE\_1](zh/nnopbase/opdev/L2_DFX_PHASE_1.md)
    -   [L2\_DFX\_PHASE\_2](zh/nnopbase/opdev/L2_DFX_PHASE_2.md)
    -   [OP\_ATTR](zh/nnopbase/opdev/OP_ATTR.md)
    -   [OP\_ATTR\_NAMES](zh/nnopbase/opdev/OP_ATTR_NAMES.md)
    -   [OP\_EMPTY\_ARG](zh/nnopbase/opdev/OP_EMPTY_ARG.md)
    -   [OP\_INPUT](zh/nnopbase/opdev/OP_INPUT.md)
    -   [OP\_MODE](zh/nnopbase/opdev/OP_MODE.md)
    -   [OP\_OUTPUT](zh/nnopbase/opdev/OP_OUTPUT.md)
    -   [OP\_OUTSHAPE](zh/nnopbase/opdev/OP_OUTSHAPE.md)
    -   [OP\_OPTION](zh/nnopbase/opdev/OP_OPTION.md)
    -   [OP\_TYPE\_REGISTER](zh/nnopbase/opdev/OP_TYPE_REGISTER.md)
    -   [OP\_WORKSPACE](zh/nnopbase/opdev/OP_WORKSPACE.md)
    -   [OpImplMode](zh/nnopbase/opdev/OpImplMode.md)
    -   [OpExecMode](zh/nnopbase/opdev/OpExecMode.md)

-   [bfloat16](zh/nnopbase/opdev/bfloat16.md)
-   [common\_types](zh/nnopbase/opdev/common_types.md)
    -   [GetStorageShape](zh/nnopbase/opdev/GetStorageShape.md)
    -   [SetStorageShape](zh/nnopbase/opdev/SetStorageShape.md)
    -   [GetOriginalShape](zh/nnopbase/opdev/GetOriginalShape.md)
    -   [SetOriginalShape](zh/nnopbase/opdev/SetOriginalShape.md)
    -   [GetViewShape](zh/nnopbase/opdev/GetViewShape.md)
    -   [SetViewShape](zh/nnopbase/opdev/SetViewShape.md)
    -   [GetStorageFormat](zh/nnopbase/opdev/GetStorageFormat.md)
    -   [SetStorageFormat](zh/nnopbase/opdev/SetStorageFormat.md)
    -   [GetOriginalFormat](zh/nnopbase/opdev/GetOriginalFormat.md)
    -   [SetOriginalFormat](zh/nnopbase/opdev/SetOriginalFormat.md)
    -   [GetViewFormat](zh/nnopbase/opdev/GetViewFormat.md)
    -   [SetViewFormat](zh/nnopbase/opdev/SetViewFormat.md)
    -   [GetViewStrides](zh/nnopbase/opdev/GetViewStrides.md)
    -   [SetViewStrides](zh/nnopbase/opdev/SetViewStrides.md)
    -   [Numel](zh/nnopbase/opdev/Numel.md)
    -   [GetDataType](zh/nnopbase/opdev/GetDataType.md)
    -   [IsEmpty](zh/nnopbase/opdev/IsEmpty.md)
    -   [SetData](zh/nnopbase/opdev/SetData.md)
    -   [SetDataType](zh/nnopbase/opdev/SetDataType.md)
    -   [SetBoolData](zh/nnopbase/opdev/SetBoolData.md)
    -   [SetIntData](zh/nnopbase/opdev/SetIntData.md)
    -   [SetFloatData](zh/nnopbase/opdev/SetFloatData.md)
    -   [SetFp16Data](zh/nnopbase/opdev/SetFp16Data.md)
    -   [SetBf16Data](zh/nnopbase/opdev/SetBf16Data.md)
    -   [CheckOverflows](zh/nnopbase/opdev/CheckOverflows.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口.md)

-   [data\_type\_utils](zh/nnopbase/opdev/data_type_utils.md)
    -   [IsBasicType](zh/nnopbase/opdev/IsBasicType.md)
    -   [IsNumberType](zh/nnopbase/opdev/IsNumberType.md)
    -   [IsRealNumberType](zh/nnopbase/opdev/IsRealNumberType.md)
    -   [TypeSize](zh/nnopbase/opdev/TypeSize.md)
    -   [ToOpDataType](zh/nnopbase/opdev/ToOpDataType.md)
    -   [ToAclDataType](zh/nnopbase/opdev/ToAclDataType.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-0.md)

-   [format\_utils](zh/nnopbase/opdev/format_utils.md)
    -   [IsPrivateFormat](zh/nnopbase/opdev/IsPrivateFormat.md)
    -   [GetPrimaryFormat](zh/nnopbase/opdev/GetPrimaryFormat.md)
    -   [ToOpFormat](zh/nnopbase/opdev/ToOpFormat.md)
    -   [ToAclFormat](zh/nnopbase/opdev/ToAclFormat.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-1.md)

-   [fp16\_t](zh/nnopbase/opdev/fp16_t.md)
-   [framework\_op](zh/nnopbase/opdev/framework_op.md)
    -   [CopyToNpu](zh/nnopbase/opdev/CopyToNpu.md)
    -   [CopyToNpuSync](zh/nnopbase/opdev/CopyToNpuSync.md)
    -   [CopyNpuToNpu](zh/nnopbase/opdev/CopyNpuToNpu.md)

-   [object](zh/nnopbase/opdev/object.md)
-   [op\_arg\_def](zh/nnopbase/opdev/op_arg_def.md)
-   [op\_cache](zh/nnopbase/opdev/op_cache.md)
-   [op\_cache\_container](zh/nnopbase/opdev/op_cache_container.md)
-   [op\_config](zh/nnopbase/opdev/op_config.md)
-   [op\_def](zh/nnopbase/opdev/op_def.md)
-   [op\_dfx](zh/nnopbase/opdev/op_dfx.md)
-   [op\_executor](zh/nnopbase/opdev/op_executor.md)
    -   [AllocTensor](zh/nnopbase/opdev/AllocTensor.md)
    -   [AllocHostTensor](zh/nnopbase/opdev/AllocHostTensor.md)
    -   [AllocIntArray](zh/nnopbase/opdev/AllocIntArray.md)
    -   [AllocFloatArray](zh/nnopbase/opdev/AllocFloatArray.md)
    -   [AllocBoolArray](zh/nnopbase/opdev/AllocBoolArray.md)
    -   [AllocTensorList](zh/nnopbase/opdev/AllocTensorList.md)
    -   [AllocScalarList](zh/nnopbase/opdev/AllocScalarList.md)
    -   [AllocScalar](zh/nnopbase/opdev/AllocScalar.md)
    -   [ConvertToTensor](zh/nnopbase/opdev/ConvertToTensor.md)
    -   [CommonOpExecutorRun](zh/nnopbase/opdev/CommonOpExecutorRun.md)
    -   [ReleaseTo](zh/nnopbase/opdev/ReleaseTo.md)
    -   [Getworkspacesize](zh/nnopbase/opdev/Getworkspacesize.md)
    -   [AbandonCache](zh/nnopbase/opdev/AbandonCache.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-2.md)

-   [op\_log](zh/nnopbase/opdev/op_log.md)
-   [platform](zh/nnopbase/opdev/platform.md)
    -   [GetCurrentPlatformInfo](zh/nnopbase/opdev/GetCurrentPlatformInfo.md)
    -   [GetSocVersion](zh/nnopbase/opdev/GetSocVersion.md)
    -   [GetSocLongVersion](zh/nnopbase/opdev/GetSocLongVersion.md)
    -   [GetBlockSize](zh/nnopbase/opdev/GetBlockSize.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-3.md)

-   [pool\_allocator](zh/nnopbase/opdev/pool_allocator.md)
-   [shape\_utils](zh/nnopbase/opdev/shape_utils.md)
    -   [ToShape](zh/nnopbase/opdev/ToShape.md)
    -   [ToShapeVector](zh/nnopbase/opdev/ToShapeVector.md)
    -   [ToContiguousStrides](zh/nnopbase/opdev/ToContiguousStrides.md)
    -   [CheckBroadcastShape](zh/nnopbase/opdev/CheckBroadcastShape.md)
    -   [BroadcastInferShape](zh/nnopbase/opdev/BroadcastInferShape.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-4.md)

-   [small\_vector](zh/nnopbase/opdev/small_vector.md)
-   [tensor\_view\_utils](zh/nnopbase/opdev/tensor_view_utils.md)
    -   [IsContiguous](zh/nnopbase/opdev/IsContiguous.md)
    -   [预留接口](zh/nnopbase/opdev/预留接口-5.md)

-   [op\_common相关接口](zh/nnopbase/opdev/op_common相关接口.md)
    -   [data\_type\_utils](zh/nnopbase/opdev/data_type_utils-6.md)
        -   [IsComplexType](zh/nnopbase/opdev/IsComplexType.md)
        -   [IsFloatingType](zh/nnopbase/opdev/IsFloatingType.md)
        -   [IsIntegralType](zh/nnopbase/opdev/IsIntegralType.md)
        -   [IsIntegralType（含bool）](zh/nnopbase/opdev/IsIntegralType（含bool）.md)
        -   [CanCast](zh/nnopbase/opdev/CanCast.md)
        -   [PromoteType](zh/nnopbase/opdev/PromoteType.md)

-   [aicpu相关接口](zh/nnopbase/opdev/aicpu相关接口.md)
    -   [AicpuTask类](zh/nnopbase/opdev/AicpuTask类.md)
    -   [AicpuTaskSpace类](zh/nnopbase/opdev/AicpuTaskSpace类.md)
    -   [aicpu\_args\_handler](zh/nnopbase/opdev/aicpu_args_handler.md)
    -   [aicpu\_ext\_info\_handle](zh/nnopbase/opdev/aicpu_ext_info_handle.md)
    -   [aicpu\_task](zh/nnopbase/opdev/aicpu_task.md)
    -   [aicpu\_uitls](zh/nnopbase/opdev/aicpu_uitls.md)

## op\_common接口

本章为CANN算子库调用提供框架公共能力，如Metadef、Log日志等。此类接口对应本项目`pkg_inc/op_common`下头文件的定义。

-   [op\_common接口列表](zh/op_common/op_common接口列表.md)
-   [log](zh/op_common/log.md)
    -   [OP\_LOGE](zh/op_common/OP_LOGE.md)
    -   [OP\_LOGD](zh/op_common/OP_LOGD.md)
    -   [OP\_LOGI](zh/op_common/OP_LOGI.md)
    -   [OP\_LOGW](zh/op_common/OP_LOGW.md)
    -   [OP\_CHECK\_IF](zh/op_common/OP_CHECK_IF.md)
    -   [OP\_CHECK\_NULL\_WITH\_CONTEXT](zh/op_common/OP_CHECK_NULL_WITH_CONTEXT.md)

-   [math](zh/op_common/math.md)
    -   [FloorDiv](zh/op_common/FloorDiv.md)
    -   [FloorAlign](zh/op_common/FloorAlign.md)
    -   [CeilDiv](zh/op_common/CeilDiv.md)
    -   [CeilAlign](zh/op_common/CeilAlign.md)
    -   [IsFloatEqual](zh/op_common/IsFloatEqual.md)

-   [platform](zh/op_common/platform.md)
    -   [GetAivCoreNum](zh/op_common/GetAivCoreNum.md)
    -   [GetAicCoreNum](zh/op_common/GetAicCoreNum.md)
    -   [GetUbSize](zh/op_common/GetUbSize.md)
    -   [GetUbBlockSize](zh/op_common/GetUbBlockSize.md)

-   [infershape](zh/op_common/infershape.md)
    -   [SetUnknownRank](zh/op_common/SetUnknownRank.md)
    -   [IsUnknownRank](zh/op_common/IsUnknownRank.md)
    -   [SetUnknownShape](zh/op_common/SetUnknownShape.md)
    -   [IsUnknownShape](zh/op_common/IsUnknownShape.md)
    -   [InferShape4Broadcast](zh/op_common/InferShape4Broadcast.md)
    -   [InferShape4Elewise](zh/op_common/InferShape4Elewise.md)
    -   [InferShape4Reduce](zh/op_common/InferShape4Reduce.md)

## build说明

[build参数说明](zh/context/build.md)