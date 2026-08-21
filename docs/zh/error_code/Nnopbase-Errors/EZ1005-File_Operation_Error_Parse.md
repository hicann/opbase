# EZ1005 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、报错原因：

```text
Failed to parse file %s. Reason: %s.
```

报错示例如下：

```text
Failed to parse file /home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/config/ascendxxx/aic-ascendxxxx-ops-info-oam.json. Reason: The operator JSON file is not in the standard key-value structure.
```

## 可能原因

1. 内置算子JSON文件被损坏。
2. 自定义算子JSON文件被损坏。

## 解决方法

1. 重新安装内置算子包。
2. 重新安装自定义算子包。
