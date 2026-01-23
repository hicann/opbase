# opbase

## 🔥Latest News

- [2025/12] 支持kirinx90，支持离线编译。
- [2025/09] opbase项目首次上线。

## 🚀概述

opbase是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库依赖的基础框架库，其提供基础的调度能力和公共依赖项，包括公共的头文件、结构体、调度框架等。关于CANN算子库请访问[ops-math](https://gitcode.com/cann/ops-math)、[ops-nn](https://gitcode.com/cann/ops-nn)、[ops-cv](https://gitcode.com/cann/ops-cv)、[ops-transformer](https://gitcode.com/cann/ops-transformer)获取算子源码实现详情。

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="320px">


## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](docs/zh/context/quick_start.md)获取简易教程，主要包括如下操作，其他文档介绍可参考[docs](docs)。

- [前提条件](docs/zh/context/quick_start.md#前提条件)：安装软件包之前，请完成基础环境搭建，包括第三方依赖等。
- [环境准备](docs/zh/context/quick_start.md#环境准备)：基础环境搭建后，需完成CANN软件包安装、环境变量配置、源码下载等。
- [编译执行](docs/zh/context/quick_start.md#编译执行)：环境准备好后，支持对源码修改（如优化、新增等），编译生成的opbase包可部署到AI业务中。
- [本地验证](docs/zh/context/quick_start.md#本地验证)：基于build.sh可执行UT用例，快速验证项目功能。

## 🔍 目录结构

关键目录如下，详细目录介绍参见[项目目录](./docs/zh/context/dir_structure.md)。

```
├── aicpu_common                   # aicpu公共函数实现
├── build.sh                       # 项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt                 # 算子编译配置文件
├── docs                           # 项目文档介绍（zh为中文，en为英文）
├── include                        # 项目公共头文件
│   └── aicpu_common               # aicpu公共函数头文件
│   └── nnopbase                   # nnopbase模块头文件
│        ├── aclnn                 # aclnn接口依赖头文件
│        └── opdev                 # 算子开发依赖头文件
│            ├── aicpu             # aicpu算子开发相关头文件
│            └── op_common         # 算子通用接口头文件
├── pkg_inc                        # 项目包间接口头文件目录
│   └── op_common                  # op_common源码头文件
├── LICENSE
├── README.md
├── requirements.txt               # 项目依赖的第三方软件包
├── scripts                        # 项目脚本文件存放目录
├── src
│   └── nnopbase                   # nnopbase代码目录
│       ├── aicpu                  # aicpu框架代码
│       ├── common                 # nnopbase公共文件
│       ├── composite_op           # 组合多算子框架代码
│       ├── individual_op          # 单一算子框架代码
│       ├── stub                   # 交叉编译场景打包脚本
│       ├── tls_guardian           # 解决glibc问题补丁
│       └── CMakeLists.txt         # nnopbase模块编译配置文件
│   └── op_common                  # op_common源码实现
│       ├── atvoss
│       ├── log
│       └── op_host
└── tests                          # 测试工程目录
```

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)