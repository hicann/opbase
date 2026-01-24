# 快速入门
## 前提条件

本项目源码编译用到的依赖如下，请注意版本要求。

- python >= 3.7.0
- gcc >= 7.3.0
- g++ >= 7.3.0
- cmake >= 3.16.0
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- patch >= 2.7（用于给第三方源码打补丁，如protobuf）
- dos2unix
- git

上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
```bash
bash install_deps.sh
```


## 环境准备

1. **安装CANN toolkit包**

    根据实际产品型号和环境架构，获取对应的`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，[下载链接](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-release/software/9.0.0/)。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **配置环境变量**

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/cann/set_env.sh
    # 指定路径安装
    # source ${install_path}/cann/set_env.sh
    ```

3. **下载源码**

    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/opbase.git
    # 安装根目录requirements.txt依赖
    pip3 install -r requirements.txt
    ```

## 编译执行
1. **编译opbase包**

    进入项目根目录，执行如下编译命令：

    > 说明：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[离线编译](build_offline.md)。

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

## 本地验证 

通过根目录的build.sh脚本执行UT或ST用例验证项目功能是否正常，build参数参见[build参数说明](./build.md#参数说明)。

> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

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