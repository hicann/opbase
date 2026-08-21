# EZ0022 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、tensor数量错误值、tensor数量正确值：

```text
Parameter %s of %s has invalid tensor num %ld. It should be %s.
```

报错示例如下：

```text
Parameter instance of Foreach has invalid tensor num 1000. It should be within the range [1, 950].
```

## 解决方法

检查输入或输出tensor的数量是否满足条件。
