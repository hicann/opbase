# EZ0002 Invalid\_Attr

## 错误信息

报错格式如下，占位符%s的含义依次为属性名、算子名或接口名、属性错误值、属性正确值：

```text
Attribute %s of %s has incorrect value %s. It should be %s.
```

报错示例如下：

```text
Attribute dim of KthValue has incorrect value 5. It should be [-3, 2].
```

## 解决方法

检查算子的属性是否正确。
