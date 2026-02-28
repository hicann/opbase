# 离线编译
离线编译是指在没有连接互联网的环境下，将软件源代码编译成可执行程序，并安装或配置到目标服务器上的过程。
本项目[编译opbase包](quick_start.md#编译执行)过程中会依赖一些开源第三方软件，这些软件联网时会自动下载，离线状态下无法直接下载。

本章提供了离线编译安装指导，在此之前请确保已按[环境准备](quick_start.md#环境准备)完成基础环境搭建。
## 获取依赖
离线编译时，需提前准备如下依赖：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| json | 3.11.3 | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| eigen | 5.0.0 | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) |
| protobuf | 25.1.0 | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |

可自行点击表格中链接下载；

或通过[third_lib_download.py](../../../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：

```bash
python ${scripts_dir}/third_lib_download.py
```
\$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

## 离线编译

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

    此时编译命令需在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件所在路径，命令如下：

    ```bash
    bash build.sh --cann_3rd_lib_path=${cann_3rd_lib_path}
    ```