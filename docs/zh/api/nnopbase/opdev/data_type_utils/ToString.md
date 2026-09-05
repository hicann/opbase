# ToString

## 功能说明

将一组数据类型序列化为字符串形式。空列表返回"[]"。`initializer_list`版非空列表返回形如"[DT_FLOAT,DT_INT8,]"的字符串，每个元素后均带英文逗号（含尾随逗号）；`array`版与`vector`版非空列表返回形如"[DT_FLOAT,DT_INT8]"的字符串，元素间以英文逗号分隔，无尾随逗号。

## 函数原型

```cpp
ge::AscendString ToString(const std::initializer_list<DataType>& dataTypes)

template <size_t N>
ge::AscendString ToString(const std::array<DataType, N>& dataTypes)

ge::AscendString ToString(const std::vector<DataType>& dataTypes)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dataTypes | 输入 | 待序列化的数据类型列表，类型为`std::initializer_list<DataType>`、`std::array<DataType, N>`或`std::vector<DataType>`。 |

## 返回值说明

返回序列化后的字符串，类型为`ge::AscendString`。空列表返回"[]"。非空列表各版本输出格式存在差异：

- `initializer_list`版：返回"[DT_FLOAT,DT_INT8,]"，每个元素后均带英文逗号（含尾随逗号）。
- `array`版与`vector`版：返回"[DT_FLOAT,DT_INT8]"，元素间以英文逗号分隔，无尾随逗号。

> **说明**：`array`/`vector`版与`initializer_list`版的输出格式不一致，属于既有设计决策：新重载修正了尾随逗号，`initializer_list`版维持原格式以避免破坏存量日志。

## 约束说明

无

## 调用示例

```cpp
// 将一组数据类型序列化为字符串并打印
void Func() {
    ge::AscendString typeStr = ToString({DT_FLOAT, DT_INT8});
    // typeStr内容为"[DT_FLOAT,DT_INT8,]"
}

// 使用std::array或std::vector序列化（无尾随逗号）
void FuncContainer() {
    ge::AscendString arrStr = ToString(std::array<ge::DataType, 2>{DT_FLOAT, DT_INT8});
    // arrStr内容为"[DT_FLOAT,DT_INT8]"

    ge::AscendString vecStr = ToString(std::vector<ge::DataType>{DT_FLOAT, DT_INT8});
    // vecStr内容为"[DT_FLOAT,DT_INT8]"
}
```
