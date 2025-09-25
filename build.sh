#!/bin/bash
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_RELATIVE_PATH="build"
BUILD_OUT="build_out"
CORE_NUMS=$(cat /proc/cpuinfo| grep "processor"| wc -l)

# print usage message
dotted_line="----------------------------------------------------------------"
usage() {
    echo "Usage:"
    echo ""
    echo "    -h, --help     Print usage"
    echo "    -v, --verbose  Display build command"
    echo "Default Build Pkg Options:"
    echo $dotted_line
    echo "    -j<N>          Set the number of threads used for building ops_base, default is 8"
    echo "    -O<N>          Compile optimization options, support [O0 O1 O2 O3], default is O2"
    echo "    --make_clean "
    echo "                   Make clean and delete related file"
    echo "    --debug        Build with debug mode"
    echo "    --cann_3rd_lib_path=<PATH>"
    echo "                   Set ascend third_party package install path, default ./third_party"
    echo "Test Options:"
    echo $dotted_line
    echo "    -u             Build and run all unit tests"
    echo "    --noexec       Only compile ut, do not execute"
    echo "    --cov          Enable code coverage for unit tests"
    echo ""
}

# parse and set options
checkopts() {
    VERBOSE=""
    THREAD_NUM=8
    ENABLE_UT="off"
    MAKE_CLEAN_ALL="off"
    EXEC_TEST="off"
    ENABLE_DEBUG="off"
    BUILD_MODE=""
    ENABLE_COVERAGE="off"
    if [[ -n "${ASCEND_HOME_PATH}" ]]; then
        echo "env exists ASCEND_HOME_PATH : ${ASCEND_HOME_PATH}"
    elif [ $UID -eq 0 ]; then
        export ASCEND_HOME_PATH=/usr/local/Ascend/latest
    else
        export ASCEND_HOME_PATH=~/Ascend/latest
    fi
    CANN_3RD_LIB_PATH="$BASEPATH/third_party"

    # Process the options
    parsed_args=$(getopt -a -o j:hvuO: -l help,verbose,cov,make_clean,debug,noexec,cann_3rd_lib_path: -- "$@") || {
    usage
    exit 1
    }

    eval set -- "$parsed_args"

    while true; do
    case "$1" in
        -h | --help)
        usage
        exit 0
        ;;
        -j)
        THREAD_NUM="$2"
        shift 2
        ;;
        -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
        -u)
        ENABLE_UT="on"
        EXEC_TEST="on"
        shift
        ;;
        -O)
        BUILD_MODE="-O$2"
        shift 2
        ;;
        --cov)
        ENABLE_COVERAGE="on"
        shift
        ;;
        --make_clean)
        MAKE_CLEAN_ALL="on"
        shift
        ;;
        --debug)
        ENABLE_DEBUG="on"
        shift
        ;;
        --noexec)
        EXEC_TEST="off"
        shift
        ;;
        --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
        --)
        shift
        break
        ;;
        *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
    done
}

mk_dir() {
    local create_dir="$1"
    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

# ops_base build start
cmake_generate_make() {
    local build_path="$1"
    local cmake_args="$2"
    if [[ "${MAKE_CLEAN_ALL}" == "on" ]];then
        echo "clear all files in build directory"
        [ -d "${build_path}" ] && rm -rf "${build_path}"
    fi
    mk_dir "${build_path}"
    cd "${build_path}"
    [ -f CMakeCache.txt ] && rm CMakeCache.txt
    [ -f Makefile ] && rm Makefile
    [ -f cmake_install.cmake ] && rm cmake_install.cmake
    [ -d CMakeFiles ] && rm -rf CMakeFiles
    echo "${cmake_args}"
    cmake ${cmake_args} ..
    if [ 0 -ne $? ]; then
        echo "execute command: cmake ${cmake_args} .. failed."
        exit 1
    fi
}

# create build path
build_ops_base() {
    echo "create build directory and build ops_base"
    cd "${BASEPATH}"

    BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
    BUILD_OUT_PATH="${BASEPATH}/${BUILD_OUT}/"
    CMAKE_ARGS="\
    -DENABLE_UT=${ENABLE_UT} \
    -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
    -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
    -DPRODUCT_SIDE=device \
    -DDCMAKE_WGET_FLAGS='--no-check-certificate' \
    -DENABLE_DEBUG=${ENABLE_DEBUG} \
    -DBUILD_MODE=${BUILD_MODE} \
    -DENABLE_COVERAGE=${ENABLE_COVERAGE}"

    cmake_generate_make "${BUILD_PATH}" "${CMAKE_ARGS}"

    make ${VERBOSE} -j${THREAD_NUM} && make package

    # make package
    if [ 0 -ne $? ]; then
        echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
        return 1
    fi
    if [ -f _CPack_Packages/makeself_staging/cann*.run ];then
        mkdir -pv $BUILD_OUT_PATH
        mv _CPack_Packages/makeself_staging/cann*.run $BUILD_OUT_PATH
    else
        echo "package ops_base run failed"
        return 1
    fi

    echo "ops_base build success!"
}

main() {
    cd "${BASEPATH}"
    checkopts "$@"
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ];then
        echo "compile thread num:$THREAD_NUM over core num:$CORE_NUMS, adjust to core num"
        THREAD_NUM=$CORE_NUMS
    fi

    g++ -v
    echo "---------------- ops_base build start ----------------"
    build_ops_base || { echo "ops_base build failed."; exit 1; }
    echo "---------------- ops_base build finished ----------------"
    if [[ "${ENABLE_UT}" == "on" && "${EXEC_TEST}" == "on" ]];then
        if [ -f "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/ut/ops_base_ut ];then
            source "${ASCEND_HOME_PATH}/bin/setenv.bash"
            export LD_LIBRARY_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"/:$LD_LIBRARY_PATH
            cd "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/ut/
            ./ops_base_ut | tee ops_base_ut.log
            if grep -q "\[  FAILED  \]" ops_base_ut.log; then
                echo "Execute ops_base_ut failed."
                exit 1
            fi
            echo "Execute ops_base_ut successful."
        else
            echo "ops_base_ut does not generated"
            exit 1
        fi
    fi
}

main "$@"