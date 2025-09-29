cmake_minimum_required(VERSION 3.16)
project(ops_base VERSION 1.0.0)
include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build ops_base with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops_base with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build ops_base with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops_base need cann source code" FALSE)
endif()

set(OPEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../open_source")
if(NOT EXISTS ${OPEN_SOURCE_DIR})
    message(STATUS "Build ops_base with dependency communtiy cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build ops_base with dependency communtiy cann pkgs installed" TRUE)
else()
    message(STATUS "Build ops_base with dependency communtiy cann pkgs not installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build ops_base with dependency communtiy cann pkgs not installed" FALSE)
endif()

include(cmake/intf_pub_linux.cmake)
include(cmake/variables.cmake)
include(cmake/config.cmake)
message(STATUS "Variables in ${CMAKE_PROJECT_NAME} project:")
cmake_print_variables(BUILD_WITH_DEPENDENCY_SOURCE)
cmake_print_variables(CANN_3RD_LIB_PATH)
cmake_print_variables(CANN_3RD_PKG_PATH)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    
    message(STATUS "ENV: $ENV{ASCEND_HOME_PATH}")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        if ("$ENV{USER}" STREQUAL "root")
            set(ASCEND_HOME_PATH /usr/local/Ascend/latest)
        else()
            set(ASCEND_HOME_PATH $ENV{HOME}/Ascend/latest)
        endif() 
    endif()
    if(ENABLE_UT)
        add_subdirectory(tests/ut)
    endif()

    include_directories(
        ${OPS_BASE_INCLUDE}
        ${OPS_BASE_INCLUDE}/op_common
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/base
        ${ASCEND_HOME_PATH}/include/toolchain
        ${ASCEND_HOME_PATH}/pkg_inc/base
    )

    add_library(ops_base SHARED ${OPS_BASE_SRC})
    target_compile_options(ops_base
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_compile_definitions(ops_base
        PRIVATE
        LOG_CPP
    )
    target_link_libraries(ops_base
        PRIVATE
        ${ASCEND_HOME_PATH}/lib64/libexe_graph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        -Wl,--no-as-needed
        ${ASCEND_HOME_PATH}/lib64/libregister.so
        -Wl,--as-needed
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libtiling_api.a
        -Wl,--no-whole-archive
        $<BUILD_INTERFACE:intf_pub_cxx17>
        ${ASCEND_HOME_PATH}/lib64/libascendalog.so
    )

    if(DEFINED ENV{TOOLCHAIN_DIR})
        set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
    else()
        if(MINRC)
            set(TOOLCHAIN_DIR /usr)
        else()
            set(TOOLCHAIN_DIR ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc)
        endif()
    endif()
    include(cmake/func.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/protobuf.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/eigen.cmake)
    add_subdirectory(aicpu_common/context)
    add_subdirectory(aicpu_common/folding)

    include(cmake/package.cmake)
else()
    message(STATUS "Build ops_base_objs only from other source code")

    add_library(ops_base_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    add_library(ops_base_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    add_library(ops_base_util_objs OBJECT ${OPS_BASE_UTIL_SRC})

    set(OPBASE_METADEF_INC
        ${METADEF_INCLUDE}
        ${METADEF_INCLUDE}/common
        ${METADEF_INCLUDE}/common/util
        ${METADEF_INCLUDE}/external
        ${METADEF_INCLUDE}/external/graph
        ${METADEF_INCLUDE}/external/exe_graph
        ${METADEF_INCLUDE}/exe_graph)

    target_include_directories(ops_base_infer_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${C_SEC_INCLUDE}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
    )
    target_compile_options(ops_base_infer_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_infer_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog)

    target_include_directories(ops_base_tiling_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${C_SEC_INCLUDE}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
    )
    target_compile_options(ops_base_tiling_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_tiling_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)

    target_include_directories(ops_base_util_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${C_SEC_INCLUDE}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
    )
    target_compile_options(ops_base_util_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_util_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)
endif()