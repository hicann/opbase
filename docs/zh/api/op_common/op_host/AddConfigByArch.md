# AddConfigByArch

## 功能说明

按NPU架构（NpuArch）批量注册算子的AI Core配置，位于`opbase`命名空间下`ArchConfigHelper`类的静态成员函数。

一次调用即可完成该架构对应的所有SoC版本的注册，无需逐个SoC调用`AddConfig`。接口内部维护架构与SoC版本的映射表（一对多），当前映射关系如下：

| NpuArch常量（Ops::Base命名空间） | 注册的SoC版本 |
| --- | --- |
| DAV_1001 | ascend910 |
| DAV_2002 | ascend310p |
| DAV_2201 | ascend910b、ascend910_93 |
| DAV_3002 | ascend310b |
| DAV_3102 | ascend610lite |
| DAV_3510 | ascend950 |

## 函数原型

```cpp
static void AddConfigByArch(ops::OpAICoreDef &aicore, NpuArch arch, ops::OpAICoreConfig &aicoreConfig);

static void AddConfigByArch(ops::OpAICoreDef &aicore, NpuArch arch);
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| aicore | 输入/输出 | 算子的AI Core定义，取OpDef实现类中`this->AICore()`的返回值。 |
| arch | 输入 | NPU架构，建议使用platform_util.h中`Ops::Base`命名空间下定义的`DAV_*`常量。 |
| aicoreConfig | 输入 | 待注册的AI Core配置，不传该参数时等价于对映射表内各SoC版本分别调用`AddConfig(soc)`。 |

## 返回值说明

无

## 约束说明

- 入参`arch`不在映射表内时，不注册任何SoC版本，接口直接返回。
- 该接口用于算子注册（OpDef构造）阶段，不适用于Tiling阶段。
- 传入`aicoreConfig`的重载中，同一`OpAICoreConfig`对象会被注册到映射表内的所有SoC版本。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
#include "op_common/op_host/util/op_arch_helper.h"

class HistogramV2 : public OpDef {
public:
    explicit HistogramV2(const char* name) : OpDef(name)
    {
        // ... 算子输入输出与属性定义 ...
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicFormatFlag(true);
        // 一次调用同时注册ascend910b与ascend910_93
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_2201, aicoreConfig);
        // 注册ascend950，使用默认配置
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510);
    }
};
```
