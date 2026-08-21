# EZ0004 Invalid\_Input

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名：

```text
Parameter %s of %s is required, but it is empty.
```

报错示例如下：

```text
Parameter input of Cumsum is required, but it is empty.
```

## 解决方法

检查算子必选参数是否设置正确。
