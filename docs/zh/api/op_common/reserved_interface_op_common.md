# 预留接口

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1** 接口列表

| 接口名称 | 功能说明 |
| --- | --- |
| GetTid() | 获取当前线程ID。 |
| GetSafeStr(T name) | 安全获取字符串指针，支持const char\*、char\*和std::string类型。 |
| GetOpInfo(const std::string\& str) | 获取算子信息字符串。 |
| GetOpInfo(const char\* str) | 获取算子信息字符串，支持空指针保护。 |
| IsContextType\<T\>() | 判断模板类型T是否为Context类型。 |
| GetOpInfo(T context) | 从TilingContext或InferShapeContext中获取算子类型和名称信息。 |
| DLOG_DEBUG | DEBUG级别日志级别常量，值为0。 |
| DLOG_INFO | INFO级别日志级别常量，值为1。 |
| DLOG_WARN | WARN级别日志级别常量，值为2。 |
| DLOG_ERROR | ERROR级别日志级别常量，值为3。 |
| OP_MODULE_ID | 算子模块ID常量，值为63。 |
| CheckLogLevel(int32_t moduleId, int32_t logLevel) | 日志级别检查函数，判断指定模块的日志级别是否使能，返回1为使能。 |
| DlogRecord(int32_t moduleId, int32_t level, const char\* fmt, ...) | 日志记录函数，按模块ID和级别记录日志。 |
| OP_LOGE_LIBOPAPI_REPORT(opName, fmt, ...) | 底层日志上报宏，记录ERROR级别日志。 |
| OP_LOGE_WITHOUT_REPORT(opName, ...) | 仅记录日志不上报错误的日志宏。 |
| OpLogSub(moduleId, level, opInfo, fmt, ...) | 指定模块和级别的日志记录宏。 |
| OpLogErrSub(moduleId, level, opInfo, fmt, ...) | 指定模块和级别的错误日志记录宏。 |
| D_OP_LOGI(opName, fmt, ...) | INFO级别日志调试宏。 |
| D_OP_LOGW(opName, fmt, ...) | WARN级别日志调试宏。 |
| D_OP_LOGE(opName, fmt, ...) | ERROR级别日志调试宏。 |
| D_OP_LOGD(opName, fmt, ...) | DEBUG级别日志调试宏。 |
| unlikely(x) | 编译器分支预测优化宏，标记x为低概率分支。 |
| likely(x) | 编译器分支预测优化宏，标记x为高概率分支。 |
| ToString(ge::DataType type) | 获取DataType对应的字符串表示。 |
| ToString(ge::Format format) | 获取Format对应的字符串表示。 |
| ToString(const gert::Shape\& shape) | 获取Shape对应的字符串表示。 |
| ToString(const std::vector\<const gert::Shape\*\>\& v) | 获取Shape指针列表对应的字符串表示。 |
| ViewErrorCode | 错误码枚举，用于infershape等场景的错误上报。 |
| LastPow2(uint64_t n) | 获取不超过n的最大的2的幂次值。 |
| MurmurHash(const void \*src, uint32_t srcLen, uint32_t seed) | Murmur哈希算法，用于计算数据的哈希值。 |
| SplitResult | 均分结果结构体，包含splitCount、splitFactor、splitTailFactor。 |
| SplitIntoEqualByParts(int64_t splitLen, int32_t parts, SplitResult\& splitResult) | 将splitLen均分为parts份，返回每份大小。 |
| SplitIntoEqualByFactor(int64_t splitLen, int32_t factor, SplitResult\& splitResult) | 按指定factor将splitLen进行均分，返回份数。 |
| GetValueToInteger\<T1, T2\>(const gert::Tensor \*constTensor, T2 \&value) | 从const Tensor中读取整数类型数据。 |
| GetConstInt\<ContextType, T\>(const ContextType \*context, int64_t inputIdx, T \&value) | 从InferShapeContext或TilingContext中获取指定输入的常量整数值。 |
| GetValueToShape\<T\>(const gert::Tensor \*constTensor, gert::Shape \&shape) | 从const Tensor中读取整数数组到Shape中。 |
| GetConstIntToShape\<ContextType\>(const ContextType \*context, int64_t inputIdx, gert::Shape \&shape) | 从InferShapeContext或TilingContext中获取指定输入的常量整数并写入Shape。 |
| IsRegbaseSocVersion(const gert::TilingParseContext\* context) | 判断当前平台是否为Regbase SoC版本。 |
| IsRegbaseSocVersion(const gert::TilingContext\* context) | 判断当前平台是否为Regbase SoC版本。 |
| EnsureNotScalar(const gert::Shape\& inShape) | 确保Shape不是标量（scalar），若为标量则返回一维Shape。 |
| AiCoreParams | AiCore参数结构体，包含ubSize、numBlocks、aicNum、l1Size、l0aSize、l0bSize、l0cSize。 |
| CompileInfoCommon | 编译信息公共结构体，包含aivNum、aicNum、ubSize、l1Size等平台信息。 |
| TilingBaseClass(gert::TilingContext\* context) | Tiling基类构造函数，接收TilingContext。 |
| DoTiling() | Tiling执行框架入口，依次调用各Tiling阶段。 |
| Reset(gert::TilingContext\* context) | 重置TilingBaseClass的context成员。 |
| IsCapable() | 纯虚函数，判断当前类是否支持该算子的Tiling。 |
| GetPlatformInfo() | 纯虚函数，获取平台信息如CoreNum、UB/L1/L0C资源大小。 |
| GetShapeAttrsInfo() | 纯虚函数，获取INPUT/OUTPUT/ATTR信息。 |
| DoOpTiling() | 纯虚函数，计算数据切分TilingData。 |
| DoLibApiTiling() | 纯虚函数，计算高阶API的TilingData。 |
| GetTilingKey() | 纯虚函数，计算TilingKey。 |
| GetWorkspaceSize() | 纯虚函数，计算Workspace大小。 |
| PostTiling() | 纯虚函数，保存Tiling数据。 |
| DumpTilingInfo() | 打印Tiling数据信息。 |
| DefaultTilingInfoDump() | 默认的Tiling数据打印实现。 |
| CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) | 计算tsch场景下的BlockDim。 |
| GetShapeDebugStr\<T\>(const T\& shape) | 获取Shape的调试字符串。 |
| GetTensorDebugStr(const gert::StorageShape\* shape, const gert::CompileTimeTensorDesc\* tensor) | 获取Tensor的调试字符串，包含dtype、shape、format信息。 |
| GetTilingContextDebugStr() | 获取TilingContext中所有输入输出Tensor的调试字符串。 |
| GetTilingDataDebugStr() | 获取TilingData的调试字符串。 |
| bfloat16 | bfloat16数据类型结构体，支持与float、double等类型的转换及算术运算。 |
| fp16_t | fp16半精度浮点数据类型结构体，支持与float、int等类型的转换及算术运算。 |
| castTraitB162B32 | B16到B32的CastTrait常量配置。 |
| castTraitB322B16 | B32到B16的CastTrait常量配置。 |
| castTraitB322Int32 | B32到Int32的CastTrait常量配置。 |
| castTraitB322Int16 | B32到Int16的CastTrait常量配置。 |
| castTraitB162Int8 | B16到Int8的CastTrait常量配置。 |
| LoadOneTensorForDtypeT\<T\>(\_\_local_mem\_\_ T \*input, MicroAPI::RegTensor\<float\> \&dst, MicroAPI::MaskReg \&preg, uint32_t offset) | 从UB加载一个对齐的Tensor到float32寄存器中。 |
| LoadTwoTensorForDtypeT\<T\>(\_\_local_mem\_\_ T \*src1, \_\_local_mem\_\_ T \*src2, MicroAPI::RegTensor\<float\> \&dst1, MicroAPI::RegTensor\<float\> \&dst2, MicroAPI::MaskReg \&dst1Preg, MicroAPI::MaskReg \&dst2Preg, uint32_t src1Offset, uint32_t src2Offset) | 从UB加载两个对齐的Tensor到float32寄存器中。 |
| StoreOneTensorForDtypeT\<T\>(\_\_local_mem\_\_ T \*output, MicroAPI::RegTensor\<float\> \&src, MicroAPI::MaskReg \&preg, uint32_t offset) | 将float32寄存器数据存储到UB中，支持bfloat16/float16/float32/int32/int16/int8/uint8类型输出。 |
| LoadUnAlignOneTensor\<T\>(\_\_local_mem\_\_ T \*&input, MicroAPI::RegTensor\<float\> \&dst, MicroAPI::UnalignReg \&uSrc, MicroAPI::MaskReg \&preg, uint32_t postUpdateStride) | 从UB加载一个非对齐的Tensor到float32寄存器中。 |
| StoreUnAlignOneTensor\<T\>(\_\_local_mem\_\_ T \*&output, MicroAPI::RegTensor\<float\> \&src, MicroAPI::UnalignReg \&uValue, MicroAPI::MaskReg \&preg, uint32_t postUpdateStride) | 将float32寄存器数据非对齐存储到UB中。 |
| CeilDiv\<T\>(T a, T b) | kernel侧向上取整除法。 |
| FloorDiv\<T\>(T a, T b) | kernel侧向下取整除法。 |
| CeilAlign\<T\>(T a, T b) | kernel侧向上对齐。 |
| FloorAlign\<T\>(T a, T b) | kernel侧向下对齐。 |
| GetUbBlockSize() | kernel侧获取UB block单元大小（32bytes）。 |
| GetVRegSize() | kernel侧获取向量寄存器大小。 |
| aclOpExecutor | acl算子执行器结构体类型。 |
| aclTensor | acl张量结构体类型。 |
| aclScalar | acl标量结构体类型。 |
| aclTensorList | acl张量列表结构体类型。 |
| \_aclCreateTensor | aclCreateTensor接口的函数指针类型。 |
| \_aclCreateScalar | aclCreateScalar接口的函数指针类型。 |
| \_aclCreateIntArray | aclCreateIntArray接口的函数指针类型。 |
| \_aclCreateFloatArray | aclCreateFloatArray接口的函数指针类型。 |
| \_aclCreateBoolArray | aclCreateBoolArray接口的函数指针类型。 |
| \_aclCreateTensorList | aclCreateTensorList接口的函数指针类型。 |
| \_aclDestroyTensor | aclDestroyTensor接口的函数指针类型。 |
| \_aclDestroyScalar | aclDestroyScalar接口的函数指针类型。 |
| \_aclDestroyIntArray | aclDestroyIntArray接口的函数指针类型。 |
| \_aclDestroyFloatArray | aclDestroyFloatArray接口的函数指针类型。 |
| \_aclDestroyBoolArray | aclDestroyBoolArray接口的函数指针类型。 |
| \_aclDestroyTensorList | aclDestroyTensorList接口的函数指针类型。 |
| ResetCacheThreadLocal | ResetCacheThreadLocal接口的函数指针类型。 |
| GET\_OP\_API\_FUNC(apiName) | 获取指定acl算子API的函数地址，并转换为对应的函数指针类型。 |
| index\_sequence\<Is...\> | 编译期索引序列结构体模板。 |
| make\_index\_sequence\_helper\<N, Is...\> | 编译期索引序列生成辅助结构体模板。 |
| make\_index\_sequence\<N\> | 生成编译期索引序列的别名模板。 |
| GetOpApiLibName() | 获取内置opapi算子库名称。 |
| GetCustOpApiLibName() | 获取自定义算子opapi库名称。 |
| GetOpApiFuncAddrInLib(void \*handler, const char \*libName, const char \*apiName) | 从指定动态库中按符号名查找函数地址。 |
| GetOpApiLibHandler(const char \*libName) | 加载指定动态库并返回句柄。 |
| GetAclnnAddrByApiName(const char \*apiName) | 遍历opapi领域库查找函数地址。 |
| GetOpApiFuncAddr(const char \*apiName) | 依次从cust\_opapi库、opapi库、领域库中查找函数地址。 |
| GetConvertType(const gert::Tensor \*ge\_tensor) | 将ge数据类型转换为acl数据类型。 |
| ConvertType(const gert::Tensor \*ge\_tensor) | 将ge Tensor转换为acl Tensor。 |
| ConvertType(std::vector\<const gert::Tensor \*\>\& ge\_tensorList) | 将ge Tensor列表转换为acl TensorList。 |
| ConvertType(T value) | 通用类型转换模板，非Tensor类型原样返回。 |
| ConvertScalarType(T value) | 将标量转换为acl Scalar。 |
| Release(aclTensor \*p) | 释放acl Tensor。 |
| Release(aclScalar \*p) | 释放acl Scalar。 |
| Release(aclTensorList \*p) | 释放acl TensorList。 |
| Release(T value) | 通用类型释放模板，非acl类型不做处理。 |
| CallRelease(Tuple t, index\_sequence) | 展开tuple并逐个调用Release释放资源。 |
| ReleaseConvertTypes(Tuple \&t) | 释放tuple中所有转换结果。 |
| ConvertTypes(Ts\&... args) | 将多个ge类型参数转换为acl类型并打包为tuple。 |
| call(Function f, Tuple t, index\_sequence) | 展开tuple参数并调用目标函数。 |
| call(Function f, Tuple t) | 展开tuple参数并调用目标函数。 |
| ConvertToOpApiFunc(params, opApiAddr, index\_sequence) | 将函数地址转换为与参数类型匹配的函数指针。 |
| ConvertToOpApiFunc(params, opApiAddr) | 将函数地址转换为与参数类型匹配的函数指针。 |
| OpApiAnyValueDeleter | 通用参数资源销毁函数指针类型。 |
| OpApiAnyValue | 通用参数容器结构体，保存参数指针及其销毁器。 |
| OpApiFunc2Stage | 两阶段Launch函数指针类型。 |
| OpApiParams | 两阶段执行参数结构体，封装转换后的参数列表、算子执行器与Launch函数。 |
| Collect(aclTensor \*p, std::vector\<OpApiAnyValue\>\& params) | 收集acl Tensor参数及其销毁器到参数列表。 |
| Collect(aclScalar \*p, std::vector\<OpApiAnyValue\>\& params) | 收集acl Scalar参数及其销毁器到参数列表。 |
| Collect(aclIntArray \*p, std::vector\<OpApiAnyValue\>\& params) | 收集acl IntArray参数及其销毁器到参数列表。 |
| Collect(aclBoolArray \*p, std::vector\<OpApiAnyValue\>\& params) | 收集acl BoolArray参数及其销毁器到参数列表。 |
| Collect(aclTensorList \*p, std::vector\<OpApiAnyValue\>\& params) | 收集acl TensorList参数及其销毁器到参数列表。 |
| Collect(T value, std::vector\<OpApiAnyValue\>\& params) | 通用参数收集模板，非acl类型不做处理。 |
| CallCollect(Tuple t, index\_sequence, std::vector\<OpApiAnyValue\>\& params) | 展开tuple并逐个调用Collect收集参数。 |
| CollectConvertedTypes(Tuple \&t, std::vector\<OpApiAnyValue\>\& params) | 收集tuple中所有转换后的参数及其销毁器。 |
