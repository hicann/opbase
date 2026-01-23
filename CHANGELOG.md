# CHANGELOG
 	 
> 本文档记录各版本的重要变更，版本按时间倒序排列。

## v8.5.0-beta.1
发布日期：2025-12-30

opbase 首个 Beta 版本 v8.5.0-beta.1 现已发布。
本版本引入了多项新增特性、问题修复及性能改进，目前仍处于测试阶段。
我们诚挚欢迎社区反馈，以进一步提升 opbase 的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/opbase/blob/master/README.md)。

### 🔗 版本地址
[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```
版本目录说明如下：
├── aarch64                 # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```
### 📌 版本配套

**CANN开源子包版本配套关系**
| CANN子包版本                         | 版本源码标签                                                 | 配套CANN版本        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------- |
| cann-opbase 8.5.0-beta.1             | [v8.5.0-beta.1](https://gitcode.com/cann/opbase/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-oam-tools   8.5.0-beta.1        | [v8.5.0-beta.1](https://gitcode.com/cann/oam-tools/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-asc-tools   8.5.0-beta.1        | [v8.5.0-beta.1](https://gitcode.com/cann/asc-tools/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-asc-devkit   8.5.0-beta.1       | [v8.5.0-beta.1](https://gitcode.com/cann/asc-devkit/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-pto-isa   8.5.0-beta.1          | [v8.5.0-beta.1](https://gitcode.com/cann/pto-isa/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-ge-compiler   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-ge-executor   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-graph-autofusion   8.5.0-beta.1 | [v8.5.0-beta.1](https://gitcode.com/cann/graph-autofusion/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-metadef   8.5.0-beta.1          | [v8.5.0-beta.1](https://gitcode.com/cann/metadef/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-dflow-executor   8.5.0-beta.1   | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-hcomm   8.5.0-beta.1            | [v8.5.0-beta.1](https://gitcode.com/cann/hcomm/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-npu-runtime   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/runtime/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 关键特性

- 【工程能力】支持离线编译。([#57](https://gitcode.com/cann/opbase/pull/57))
- 【资料优化】新增离线编译说明文档。([#56](https://gitcode.com/cann/opbase/pull/56))

### 🐛 问题修复
- 现场机器离线状态下进行opbase编译报错。([Issue6](https://gitcode.com/cann/opbase/issues/6))
- obpase包编译报错 non-constant-expression cannot be narrowed。([Issue22](https://gitcode.com/cann/opbase/issues/22))