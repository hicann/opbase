# EZ0003 Invalid\_Attr\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为属性名、算子名或接口名、大小错误值、大小正确值：

```text
Attribute %s of %s has incorrect size %s. It should be %s.
```

报错示例如下：

```text
Attribute strides of ExtendConvTranspose has incorrect size 2. It should be 3. 
```

## 解决方法

检查算子的属性是否正确。
