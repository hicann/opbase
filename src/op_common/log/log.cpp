/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log.cpp
 * \brief
 */

#include <string>
#include <vector>
#include "op_common/log/log.h"
#include <graph/utils/type_utils.h>

namespace Ops {
namespace Base {
std::string ToString(ge::DataType type)
{
    return ge::TypeUtils::DataTypeToSerialString(type);
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(ge::Format format)
{
    return ge::TypeUtils::FormatToSerialString(format);
}

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

template <typename T>
static std::string ToString(const std::vector<T>& v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

std::string ToString(const std::vector<const gert::Shape*>& v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << ToString(ToVector(*v[i])) << ", ";
        }
        oss << ToString(ToVector(*v[v.size() - 1]));
    }
    oss << "]";
    return oss.str();
}

std::string ToString(const gert::Shape& shape)
{
    return ToString(ToVector(shape));
}

#ifdef BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG

const static std::string g_msg = R"(
{
  "error_info_list": [
    {
      "errClass": "Vector Operator Plugin Errors",
      "errTitle": "ONNX_Model_Data_Error",
      "ErrCode": "E76002",
      "ErrMessage": "No ONNX parser is registered for optype [%s].",
      "Arglist": "optype",
      "suggestion": {
        "Possible Cause": "No parser is registered for the optype in ONNX model conversion.",
        "Solution": "Submit an issue to request for support at https://gitee.com/ascend, or remove this type of operators from your model."
      }
    },
    {
      "ErrCode": "E70001",
      "errTitle": "Invalid_Argument_Input_Tensor_Missing",
      "ErrMessage": "In op[%s], the mandatory parameter[%s] is missed.",
      "Arglist": "op_name,param_name"
    },
    {
      "ErrCode": "E70002",
      "errTitle": "Not_Supported_Format",
      "ErrMessage": "In op[%s], the format of input[%s] should be one of [%s], but actually is [%s].",
      "Arglist": "op_name,param_name,expected_format_list,format"
    },
    {
      "ErrCode": "E70003",
      "errTitle": "Not_Supported_Format",
      "ErrMessage": "In op[%s], the dtype of input[%s] should be one of [%s], but actually is [%s].",
      "Arglist": "op_name,param_name,expected_data_type_list,data_type"
    },
    {
      "ErrCode": "E70004",
      "errTitle": "Not_Supported_Data_Type",
      "ErrMessage": "In op[%s], the parameter[%s]'s type should be [%s],  but actually is [%s].",
      "Arglist": "op_name,param_name,param_type,actual_type"
    },
    {
      "ErrCode": "E70005",
      "ErrMessage": "Op[%s] get attr[%s] failed.",
      "Arglist": "op_name,param_name"
    },
    {
      "ErrCode": "E70006",
      "ErrMessage": "Op[%s] set attr[%s] failed.",
      "Arglist": "op_name,param_name"
    },
    {
      "ErrCode": "E70007",
      "errTitle": "NOT_SUPPORTED",
      "ErrMessage": "The op[%s] input parameter[%s] should be [%s], actual the input is [%s].",
      "Arglist": "op_name,param_name,excepted_value,input_value"
    },
    {
      "ErrCode": "E70008",
      "ErrMessage": "The op [%s] update [%s] param failed.",
      "Arglist": "op_name,param_name"
    },
    {
      "ErrCode": "E70009",
      "ErrMessage": "The op [%s] has rule: %s, but [%s] is [%s].",
      "Arglist": "op_name,rule_desc,param_name,param_value"
    },
    {
      "ErrCode": "E70010",
      "errTitle": "Not_Supported_Shape",
      "ErrMessage": "In op[%s], the shape size(product of all dimensions) of input[%s] should less than [%s], but actually is [%s].",
      "Arglist": "op_name,input_name,max_value,real_value"
    },
    {
      "ErrCode": "E70011",
      "errTitle": "Not_Supported_Shape",
      "ErrMessage": "In op[%s], the num of dimensions of input[%s] should be in the range of [%s, %s], but actually is [%s].",
      "Arglist": "op_name,param_name,max_value,min_value,real_value"
    },
    {
      "ErrCode": "E70012",
      "errTitle": "Invalid_Argument_Shape_Mismatch",
      "ErrMessage": "In op[%s], the inputs[%s][%s] could not be broadcast together with shapes[%s][%s].",
      "Arglist": "op_name,input1_name,input2_name,input1_shape,input2_shape"
    },
    {
      "ErrCode": "E70013",
      "errTitle": "Not_Supported_Data_Type",
      "ErrMessage": "In op[%s], the dtype of inputs[%s][%s] should be same, but actually is [%s][%s].",
      "Arglist": "op_name,input1_name,input2_name,input1_dtype,input2_dtype"
    },
    {
      "ErrCode": "E70014",
      "errTitle": "Invalid_Argument_Shape_Mismatch",
      "ErrMessage": "In op[aipp], aipp output H and W should be equal to data's H and W, but actually aipp output H is [%s], aipp output W is [%s], data's H is [%s], data's W is [%s].",
      "Arglist": "aipp_output_H,aipp_output_W,data_H,data_W"
    },
    {
      "ErrCode": "E70015",
      "errTitle": "Not_Supported_Shape",
      "ErrMessage": "In op[%s], the shape of input[%s] is invalid, [%s].",
      "Arglist": "op_name,param_name,error_detail"
    },
    {
      "ErrCode": "E70016",
      "errTitle": "Not_Supported_Shape",
      "ErrMessage": "In op[%s], the shape of inputs[%s][%s] are invalid, [%s].",
      "Arglist": "op_name,param_name1,param_name2,error_detail"
    },
    {
      "ErrCode": "E70017",
      "errTitle": "Not_Supported_Shape",
      "ErrMessage": "In op[%s], the shape of output[%s] is invalid, [%s].",
      "Arglist": "op_name,param_name,error_detail"
    },
    {
      "ErrCode": "E70018",
      "errTitle": "Inner_Error_Compile_Fail",
      "ErrMessage": "In op[%s], failed to get compileparams, mainly get [%s] error.",
      "Arglist": "op_name,param_name"
    },
    {
      "errClass": "Vector Errors",
      "errTitle": "Invalid_Config_Value",
      "ErrCode": "E80001",
      "ErrMessage": "The %s information is missing in the %s configuration file of OPP package",
      "Arglist": "config_name, file_name",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "try to reinstall the OPP package."
      }
    },
    {
      "errClass": "Public Operator Errors",
      "errTitle": "Unsupported_Operator",
      "ErrCode": "EZ0501",
      "ErrMessage": "IR for Op [%s, optype [%s]], is not registered.",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "IR for the operator type is not registered.",
        "Solution": "Submit an issue to request for support at https://gitee.com/ascend, or remove this type of operators from your model."
      }
    },
    {
      "errClass": "Public Operator Errors",
      "errTitle": "Unsupported_Operator",
      "ErrCode": "EZ3002",
      "ErrMessage": "Optype [%s] of Ops kernel [%s] is unsupported. Reason: %s.",
      "Arglist": "optype,opskernel,reason",
      "suggestion": {
        "Possible Cause": "The operator type is unsupported in the operator information library due to specification mismatch.",
        "Solution": "Submit an issue to request for support at https://gitee.com/ascend, or remove this type of operators from your model."
      }
    },
    {
      "errClass": "Public Operator Errors",
      "errTitle": "Unsupported_Operator",
      "ErrCode": "EZ3003",
      "ErrMessage": "No supported Ops kernel and engine are found for [%s], optype [%s].",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "The operator is not supported by the system. Therefore, no hit is found in any operator information library.",
        "Solution": "1. Check that the OPP component is installed properly. 2. Submit an issue to request for the support of this operator type."
      }
    },
    {
      "errClass": "Public Operator Errors",
      "errTitle": "Unsupported_Operator",
      "ErrCode": "EZ9010",
      "ErrMessage": "No parser is registered for Op [%s, optype [%s]].",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "No parser is registered for the operator type.",
        "Solution": "Submit an issue to request for support at https://gitee.com/ascend."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Input_Shape",
      "ErrCode": "EZ0001",
      "ErrMessage": "OP[%s] %sth input has incorrect shape[: %s], it should be[: %s].",
      "Arglist": "op_name, index, incorrect_shape, correct_shape",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check whether the input shape is correct.\n2. Dump GE Graph, find error node in graph and check the shape."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Attr",
      "ErrCode": "EZ0002",
      "ErrMessage": "OP[%s] attr[: %s], has incorrect value[: %s], it should be[: %s].",
      "Arglist": "op_name, attr_name, incorrect_val, correct_val",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check the value of the attribute in this api."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Attr_Size",
      "ErrCode": "EZ0003",
      "ErrMessage": "OP[%s] attr[: %s], has incorrect size[: %s], it should be[: %s].",
      "Arglist": "op_name, attr_name, incorrect_size, correct_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check the size of the attribute in this api"
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Input",
      "ErrCode": "EZ0004",
      "ErrMessage": "OP[%s] get [%s] failed.",
      "Arglist": "op_name, param_name",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check the input information of the operator.\n2. Dump GE Graph and check the input in graph."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Input_ShapeSize",
      "ErrCode": "EZ0005",
      "ErrMessage": "OP[%s] %sth input has incorrect shape size[: %s], it should be[: %s].",
      "Arglist": "op_name, index, incorrect_size, correct_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check input dimensions.\n2. Dump GE Graph, find error node in graph and check input dimension."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Input_Format",
      "ErrCode": "EZ0006",
      "ErrMessage": "OP[%s] %s has incorrect format[: %s], it should be %s.",
      "Arglist": "op_name, param_name, data_format, expected_format_list",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check input format.\n2. Dump GE Graph, find error node in graph and check input format."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Input_Dtype",
      "ErrCode": "EZ0007",
      "ErrMessage": "OP[%s] %s has incorrect dtype[: %s], it should be %s.",
      "Arglist": "op_name, param_name, data_dtype, expected_dtype_list",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check input dtype.\n2. Dump GE Graph, find error node in graph and check input dtype."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape",
      "ErrCode": "EZ0008",
      "ErrMessage": "OP %s's parameter %s has incorrect shape [%s], it should be [%s].",
      "Arglist": "op_name, param_name, incorrect_shape, correct_shape",
      "suggestion": {
        "Possible Cause": "Parameter tensor shape does not match expected shape.",
        "Solution": "Check whether the shape of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape",
      "ErrCode": "EZ0009",
      "ErrMessage": "OP %s's parameter %s has incorrect shape [%s]. Reason: %s.",
      "Arglist": "op_name, param_name, incorrect_shape, reason",
      "suggestion": {
        "Possible Cause": "Parameter tensor shape validation failed.",
        "Solution": "Check whether the shape of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape",
      "ErrCode": "EZ0010",
      "ErrMessage": "OP %s's parameters %s have incorrect shapes %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_shapes, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensors have shape validation failures.",
        "Solution": "Check whether the shapes of input/output tensors meet the requirements."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape_Dim",
      "ErrCode": "EZ0011",
      "ErrMessage": "OP %s's parameter %s has incorrect shape dim %s, it should be %s.",
      "Arglist": "op_name, param_name, incorrect_dim, correct_dim",
      "suggestion": {
        "Possible Cause": "Parameter tensor dimension does not match expected dimension.",
        "Solution": "Check whether the shape dimension of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape_Dim",
      "ErrCode": "EZ0012",
      "ErrMessage": "OP %s's parameter %s has incorrect shape dim %s. Reason: %s.",
      "Arglist": "op_name, param_name, incorrect_dim, reason",
      "suggestion": {
        "Possible Cause": "Parameter tensor dimension validation failed.",
        "Solution": "Check whether the shape dimension of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape_Dim",
      "ErrCode": "EZ0013",
      "ErrMessage": "OP %s's parameters %s have incorrect shape dims %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_dims, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensors have dimension validation failures.",
        "Solution": "Check whether the shape dimensions of input/output tensors meet the requirements."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape_Size",
      "ErrCode": "EZ0014",
      "ErrMessage": "OP %s's parameter %s has incorrect shape size %s, it should be %s.",
      "Arglist": "op_name, param_name, incorrect_size, correct_size",
      "suggestion": {
        "Possible Cause": "Parameter tensor size does not match expected size.",
        "Solution": "Check whether the shape size of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Shape_Size",
      "ErrCode": "EZ0015",
      "ErrMessage": "OP %s's parameters %s have incorrect shape sizes %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_sizes, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensors have size validation failures.",
        "Solution": "Check whether the shape sizes of input/output tensors meet the requirements."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Format",
      "ErrCode": "EZ0016",
      "ErrMessage": "OP %s's parameter %s has incorrect format %s, it should be %s.",
      "Arglist": "op_name, param_name, incorrect_format, correct_format",
      "suggestion": {
        "Possible Cause": "Parameter tensor format does not match expected format.",
        "Solution": "Check whether the format of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Format",
      "ErrCode": "EZ0017",
      "ErrMessage": "OP %s's parameters %s have incorrect formats %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_formats, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensors have format validation failures.",
        "Solution": "Check whether the formats of input/output tensors meet the conditions."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Dtype",
      "ErrCode": "EZ0018",
      "ErrMessage": "OP %s's parameter %s has incorrect dtype %s, it should be %s.",
      "Arglist": "op_name, param_name, incorrect_dtype, correct_dtype",
      "suggestion": {
        "Possible Cause": "Parameter tensor data type does not match expected dtype.",
        "Solution": "Check whether the dtype of input/output tensor is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Dtype",
      "ErrCode": "EZ0019",
      "ErrMessage": "OP %s's parameter %s has incorrect dtype %s. Reason: %s.",
      "Arglist": "op_name, param_name, incorrect_dtype, reason",
      "suggestion": {
        "Possible Cause": "Parameter tensor data type validation failed.",
        "Solution": "Check whether the dtype of input/output tensor meets the conditions."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument_Tensor_Dtype",
      "ErrCode": "EZ0020",
      "ErrMessage": "OP %s's parameters %s have incorrect dtypes %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_dtypes, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensors have data type validation failures.",
        "Solution": "Check whether the dtypes of input/output tensors meet the conditions."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "EZ0021",
      "ErrMessage": "OP %s's parameter %s has invalid tensor num %ld, it should be %s.",
      "Arglist": "op_name, param_name, incorrect_num, correct_num",
      "suggestion": {
        "Possible Cause": "Parameter tensor count does not match expected number.",
        "Solution": "Check whether the tensor number of input tensorlist meets the conditions."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "EZ0022",
      "ErrMessage": "OP %s's parameters %s have invalid tensor nums %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_nums, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter tensor count validations failed.",
        "Solution": "Check whether the tensor numbers of input tensorlists meet the conditions."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "EZ0023",
      "ErrMessage": "OP %s's parameter %s has incorrect value %s. Reason: %s.",
      "Arglist": "op_name, param_name, incorrect_value, reason",
      "suggestion": {
        "Possible Cause": "Parameter value validation failed.",
        "Solution": "Check whether the parameter value is correct."
      }
    },
    {
      "errClass": "Operator Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "EZ0024",
      "ErrMessage": "OP %s's parameters %s have incorrect values %s. Reason: %s.",
      "Arglist": "op_name, param_names, incorrect_values, reason",
      "suggestion": {
        "Possible Cause": "Multiple parameter value validations failed.",
        "Solution": "Check whether the parameter values meet the conditions."
      }
    }
  ]
}
)";
REG_FORMAT_ERROR_MSG(g_msg.c_str(), g_msg.length());

#endif
} // namespace Base
} // namespace Ops