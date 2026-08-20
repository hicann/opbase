# EZ1004 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、报错原因：

```text
Failed to parse file %s. Reason: %s.
```

报错示例如下：

```text
Failed to parse file /home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe//kernel/ascendxxxx/ops_legacy/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.json. Reason: [json.exception.parse_error.101] parse error at line 4, column 14: syntax error while parsing object - unexpected string literal; expected '}'.
```

## 解决方法

需按照Reason中的提示定位问题，提供正确的文件。
