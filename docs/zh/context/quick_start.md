# 快速入门
## 源码编译

###  第三方软件依赖

本项目编译过程依赖的第三方开源软件列表如下：

| 开源软件   | 版本       | 下载地址                                                     |
| ---------- | ---------- | ------------------------------------------------------------ |
| json       | 3.11.3     | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| makeself   | 2.5.0      | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| eigen      | 5.0.0      | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) |
| protobuf   | 25.1.0     | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |

若您的编译环境可以访问网络，请参考[联网编译](#联网编译)，编译脚本会自动联网下载第三方软件。否则，请参考[未联网编译](#未联网编译)手动下载第三方软件。

编译命令通过根目录**build.sh**实现，`bash build.sh --help`命令可查看所有功能参数，详细介绍参考[build参数说明](./build.md#参数说明)。

### 联网编译

1. **编译opbase包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh
    ```
    若提示如下信息，则说明编译成功。
    
    ```bash
    Self-extractable archive "cann-opbase_${cann_version}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存储于build_out目录下。

2. **安装opbase包**
   
    ```bash
    ./cann-opbase_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}表示指定安装路径，若不指定，默认安装路径为：`/usr/local/Ascend`；若指定，一般安装在\$\{install\_path\}目录下。

3. **（可选）卸载opbase包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/opbase/script/uninstall.sh
    ```

### 未联网编译

若在没有连接互联网的环境下编译，需要提前准备好依赖的第三方软件，再进行源码编译。具体过程如下：

1. **检查基础环境是否完备**

    请确保已完成[环境准备](#环境准备)，包括CANN包安装、源码下载等。

    - 在联网环境中，进入[本项目主页](https://gitcode.com/cann/ops-cv)，通过`下载ZIP`或`clone`按钮，根据指导完成源码下载。
    - 连接离线环境，上传源码至您指定的目录下。若下载的是源码压缩包，请先进行解压。

2. **下载第三方软件依赖**

    在联网环境中提前下载第三方软件，目前有如下方式，请按需选择：
    
    - 方式1：根据[第三方软件依赖](#第三方软件依赖)提供的表格手动下载，若从其他地址下载，注意版本号一致。
      
    - 方式2：通过[third_lib_download.py](../../../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：
      
        ```bash
        python ${scripts_dir}/third_lib_download.py
        ```
        \$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

3. **编译算子包**

   将下载好的第三方软件上传至离线环境，可存放在`third_party`目录或自定义目录下。**推荐前者，其编译命令与联网编译场景下的命令一致。**

    - **third\_party目录**（推荐）
      
      请在本项目根目录创建`third_party`目录（若有则无需创建），将第三方软件拷贝到该指定目录。此时编译命令与联网编译命令一致：
      
      ```bash
      bash build.sh
      ```
      
    - **自定义目录**

      在离线环境的任意位置新建`${cann_3rd_lib_path}`目录，将第三方软件拷贝到该目录，请确保该目录有权限访问。

        ```bash
      mkdir -p ${cann_3rd_lib_path}
        ```

      此时在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件路径。假设路径为`/path/cann_3rd_lib_path`，编译命令如下：

      ```bash
      bash build.sh --cann_3rd_lib_path=${cann_3rd_lib_path}
      # bash build.sh --cann_3rd_lib_path=/path/cann_3rd_lib_path
      ```

4. **安装/卸载算子包**
   
    未联网和联网场景下编译得到算子包结果一样，默认存放于项目根目录build_out目录下，并且安装和卸载的操作命令也一样，具体参见[联网编译](#联网编译)。


## 本地验证 

> **说明**：
>
> - 命令通过根目录**build.sh**实现，`bash build.sh --help`命令可查看所有功能参数，详细介绍参考[build参数说明](./build.md#参数说明)。
> - 执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

### 执行UT
```bash
# 方式1: 编译并执行所有的UT测试用例
bash build.sh -u
# 方式2: 编译所有的UT测试用例但不执行
bash build.sh -u --noexec
# 方式3：执行UT并查看覆盖率
bash build.sh -u --cov
```

以编译并执行所有的UT测试为例，执行上述命令后出现如下内容，表示执行成功
```bash
bash build.sh -u
```

执行完成后出现如下内容，表示执行成功。
```bash
Global Global test environment tear-down
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
Execute ops_base_ut successful.
```

### 执行ST
```bash
# 方式1: 编译并执行所有的ST测试用例
bash build.sh -s
# 方式2: 编译所有的ST测试用例但不执行
bash build.sh -s --noexec
# 方式3：执行ST并查看覆盖率
bash build.sh -s --cov
```

以编译并执行所有的UT测试为例，执行上述命令后出现如下内容，表示执行成功
```bash
bash build.sh -s
```

执行完成后出现如下内容，表示执行成功。
```bash
Global Global test environment tear-down
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
Execute ops_base_st successful.
```

其中\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。