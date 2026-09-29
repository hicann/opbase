# AddDirectInvokeTask

通过三个生命周期回调将计算任务加入 executor。头文件为 `opdev/direct_invoke_task.h`。

```cpp
aclnnStatus op::AddDirectInvokeTask(
    const char* l0Name, uint32_t opType,
    DirectInvokePrepareFn prepare,
    DirectInvokeLaunchFn launch,
    DirectInvokeDestroyFn destroy,
    aclOpExecutor* executor, OpArgContext* args);
```

## 回调与生命周期

- `prepare(args, workspaceRequest, launchState)` 在注册时同步执行。实现选择、tiling、工作区大小等每次调用的输入均来自 `args`。算子专属回调可以引用不可变的实现元数据。
- `launch(launchState, args, stream)` 在执行时提交计算，可包含多个设备任务。最终输入、输出和工作区地址从本次 `args` 读取。
- `destroy(launchState)` 在注册失败或 launcher 销毁时释放非空状态，包含 prepare 失败返回的部分状态；每个状态恰好释放一次。无状态任务可返回空状态，此时不调用 destroy。

三个回调都必须非空且为 `noexcept`。opbase 将 workspace 请求初始化为 `{nullptr, 0, 0}`、状态初始化为空。
`DirectInvokeWorkspaceRequest` 的每个大小对应一个 executor 分配的工作区 tensor，由 opbase 负责对齐和内存复用。
大小数组必须在注册返回前保持有效，例如由 launchState 持有，不能指向 prepare 栈变量。

`args` 必须由 `GetOpArgContext` 创建，所有注册返回路径都由 opbase 消费。
允许空的可选输入 tensor，输出不能空；入口的 workspace 参数列表必须为空或不存在。
回调模块应覆盖 executor 的生命周期，异步设备任务使用的代码和资源还必须存活到设备执行完成；destroy 不是设备完成通知。

## 计算、缓存与 DFX 契约

任务统一为计算任务，内部使用与设备引擎无关的 `DIRECT_INVOKE` 分类，不声明 AiCore/AiCpu/Dvpp。
opbase 保留输入输出 dump、适用的 overflow 检查，不在回调边界上报 profiling；具体设备任务 profiling 由实际下发路径处理。

DirectInvoke 使用准备状态复用：同一 executor 设置为 repeatable 后，prepare 只执行一次，
每次执行都调用 producer 的 launch，传入同一 launchState、本次 args 和 stream。
创建新的 executor 会重新 prepare，不提供跨 executor 的 prepare 结果缓存。

opbase 内部禁止包含 DirectInvoke 的 executor 使用设备任务缓存重放。producer 的直接 runtime
下发不填充 OpExecCache 任务队列，不能通过重放该队列绕过 launch；producer 无需提供额外缓存接口。

- 每次执行的 tensor/workspace 地址从 args 读取，launchState 不保存待重绑定的地址。
- 固定实现和执行环境下，准备结果只依赖 args；静态元数据在 executor 存活期间保持不变。
- 动态输出形状及现有框架限制仍可禁止 repeat，注册不会重新启用已禁止的复用。
- destroy 在 executor 销毁时释放状态，不能假设每次 launch 后立即执行。

## 接口迁移

原任务描述结构、taskConfig、引擎类型和 producer 缓存策略已移除。
调用方直接传入三个回调，原先通过 taskConfig 传入的静态 descriptor 可由算子专属 prepare 回调引用；每次调用变化的数据应编码为常规 `args` 参数。
本次为不兼容接口变更，调用方及 opbase 必须一起重新编译，不保留旧接口别名。
