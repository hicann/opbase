# AddConfigByArch

## Function

Registers AI Core configurations for an operator in batches by NPU architecture (NpuArch). It is a static member function of the `ArchConfigHelper` class in the `opbase` namespace.

A single call registers all SoC versions corresponding to the architecture, without calling `AddConfig` for each SoC. A mapping table (one-to-many) between architectures and SoC versions is maintained inside the interface. The current mappings are as follows:

| NpuArch constant (Ops::Base namespace) | Registered SoC versions |
| --- | --- |
| DAV_1001 | ascend910 |
| DAV_2002 | ascend310p |
| DAV_2201 | ascend910b, ascend910_93 |
| DAV_3002 | ascend310b |
| DAV_3102 | ascend610lite |
| DAV_3510 | ascend950 |

## Prototype

```cpp
static void AddConfigByArch(ops::OpAICoreDef &aicore, NpuArch arch, ops::OpAICoreConfig &aicoreConfig);

static void AddConfigByArch(ops::OpAICoreDef &aicore, NpuArch arch);
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| aicore | Input/Output| AI Core definition of the operator, that is, the return value of `this->AICore()` in the OpDef implementation class.|
| arch | Input| NPU architecture. You are advised to use the `DAV_*` constants defined in the `Ops::Base` namespace in platform_util.h.|
| aicoreConfig | Input| AI Core configuration to register. If this parameter is not transferred, it is equivalent to calling `AddConfig(soc)` for each SoC version in the mapping table.|

## Returns

None

## Restrictions

- If `arch` is not in the mapping table, no SoC version is registered and the interface returns directly.
- This interface applies to the operator registration phase (OpDef construction) only, not the tiling phase.
- In the overload with `aicoreConfig` transferred, the same `OpAICoreConfig` object is registered for all SoC versions in the mapping table.

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
#include "op_common/op_host/util/op_arch_helper.h"

class HistogramV2 : public OpDef {
public:
    explicit HistogramV2(const char* name) : OpDef(name)
    {
        // ... Input, output, and attribute definitions of the operator ...
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicFormatFlag(true);
        // A single call registers both ascend910b and ascend910_93
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_2201, aicoreConfig);
        // Registers ascend950 with the default configuration
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510);
    }
};
```
