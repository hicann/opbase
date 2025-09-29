project(ops_base_ut)

file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
include_directories(
    ${OPS_BASE_INCLUDE}
    ${OPS_BASE_INCLUDE}/op_common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/base
    ${ASCEND_HOME_PATH}/include/toolchain
    ${ASCEND_HOME_PATH}/pkg_inc/base
)

add_executable(ops_base_ut ${UT_SOURCES})

target_compile_options(ops_base_ut PUBLIC
    -fPIE
    -g -O0
    )
if(ENABLE_COVERAGE)
    target_compile_options(ops_base_ut PRIVATE
    --coverage
    -fprofile-arcs
    -ftest-coverage)
endif()
target_compile_definitions(ops_base_ut
    PRIVATE
    LOG_CPP
    )

target_link_libraries(ops_base_ut
    PRIVATE
    ops_base
    gtest
    gtest_main
    $<BUILD_INTERFACE:intf_pub_cxx17>
    ${ASCEND_HOME_PATH}/lib64/libascendalog.so
    $<$<BOOL:${ENABLE_COVERAGE}>:gcov>
    )

