/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_OPDEV_DIRECT_INVOKE_TASK_H_
#define OP_API_COMMON_INC_OPDEV_DIRECT_INVOKE_TASK_H_

#include <cstdint>
#include "acl/acl_base_rt.h"
#include "aclnn/acl_meta.h"

struct aclOpExecutor;

namespace op {
struct OpArgContext;

/**
 * One executor-allocated workspace tensor per byte size. opbase handles
 * alignment and memory reuse. sizes must remain valid until AddDirectInvokeTask
 * returns (for example, owned by launchState); opbase consumes it synchronously
 * after prepare returns and does not retain the pointer.
 */
struct DirectInvokeWorkspaceRequest {
    const uint64_t* sizes;
    uint32_t count;
    uint32_t reserved;
};

/**
 * Prepare one executor-local compute task in ACLNN phase 1.
 * All per-call inputs to implementation selection, tiling and workspace sizing
 * must come from args. An operator-specific callback may reference immutable
 * implementation metadata. For a fixed implementation and execution environment,
 * equivalent args must yield equivalent preparation results. Do not hide dynamic
 * configuration in globals, opaque pointer attributes or tensor device contents.
 *
 * opbase initializes workspaceRequest to {nullptr, 0, 0} and launchState to null.
 * A non-null state returned on success or failure is destroyed exactly once.
 * State must be address-independent: obtain final tensor/workspace addresses from
 * launch's args. Copy any metadata retained in launchState; do not retain pointers
 * into args, tensor descriptors or attribute containers. Stateless tasks may
 * return null; destroy is then not called. Like AI Core tiling and launch
 * configuration, state is a reusable preparation result, not per-execution data.
 */
using DirectInvokePrepareFn = aclnnStatus (*)(const OpArgContext* args, DirectInvokeWorkspaceRequest* workspaceRequest,
                                              void** launchState) noexcept;

/**
 * Submit compute work in ACLNN phase 2 using the caller's stream. Multiple device
 * dispatches are allowed, with producer-managed dependencies for any other stream.
 * On an explicitly repeatable executor, prepare runs once and the same launchState
 * is reused, but this callback is invoked on every execution. Read all current
 * tensor/workspace addresses from args and use the stream supplied for that run.
 * OpExecCache records this callback and owns an argument snapshot. Cache hits
 * reuse preparation, rebind final addresses and invoke launch on the supplied
 * stream; producers need not record individual runtime dispatches.
 * For reusable tasks, treat argument metadata as read-only. Consume addresses
 * during this call; do not retain pointers into args after launch returns. State must not
 * make the computation depend on the number of previous launches. As with an
 * AI Core cache entry, concurrent execution of the same entry is not supported.
 *
 * opbase dumps inputs/outputs and performs applicable overflow checks around
 * the callback. It reports no profiling at the callback boundary, does not
 * infer device engine types, and does not report device-task profiling on
 * behalf of the actual dispatches.
 */
using DirectInvokeLaunchFn = aclnnStatus (*)(void* launchState, const OpArgContext* args, aclrtStream stream) noexcept;

/**
 * Destroy a non-null state on registration failure or after its last launcher/cache owner, including
 * partially prepared state. This is not a device-completion notification: producers
 * must retain code images and resources until submitted asynchronous work finishes,
 * even when launch reports an error. opbase does not synchronize or unload modules.
 * Cache eviction may call destroy on the cache GC thread. Destruction must not
 * depend on the registering thread's TLS or an implicitly current device context.
 */
using DirectInvokeDestroyFn = void (*)(void* launchState) noexcept;

/**
 * Attach a compute task by registering three required, non-throwing callbacks.
 * prepare runs synchronously; launch/destroy and their modules must remain valid
 * for the executor/cache lifetime (and submitted work's lifetime where applicable).
 *
 * args is consumed on every return path and must be created by GetOpArgContext.
 * Optional input tensors may be null; outputs must be non-null. The entry workspace
 * list must be absent or empty; prepare requests buffers that opbase appends to args.
 *
 * Profiling and dump attribution comes from opType and the enclosing DFX
 * context captured at registration time, same as the AI Core path. l0Name is
 * reserved and currently unused, mirroring CreatAiCoreKernelLauncher; callers
 * wanting attributed profiling must register under the framework DFX macros
 * (L2_DFX_PHASE_1/L0_DFX).
 *
 * opbase controls cache eligibility internally. An eligible L2 cache hit skips
 * preparation and replays launch with cache-owned args and shared prepared state.
 * Explicit executor repeat also calls launch each time. Dynamic output shapes,
 * unsupported argument rebinding and existing DFX restrictions still limit reuse.
 * State destruction may be deferred until cache eviction; callback modules and
 * immutable implementation metadata must remain valid for that lifetime.
 * As with AI Core, the L2 cache key must cover all per-call preparation inputs;
 * DirectInvoke does not create another cache key or implementation registry.
 * This API replaces the former descriptor-based interface; rebuild callers and
 * opbase together. No compatibility alias is provided for the old ABI.
 */
aclnnStatus AddDirectInvokeTask([[maybe_unused]] const char* l0Name, uint32_t opType, DirectInvokePrepareFn prepare,
                                DirectInvokeLaunchFn launch, DirectInvokeDestroyFn destroy, aclOpExecutor* executor,
                                OpArgContext* args);

} // namespace op

#endif // OP_API_COMMON_INC_OPDEV_DIRECT_INVOKE_TASK_H_
