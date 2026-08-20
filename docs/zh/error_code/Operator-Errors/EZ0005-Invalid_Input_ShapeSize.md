# EZ0005 Invalid\_Input\_ShapeSize

## 错误信息

报错格式如下，占位符%s的含义依次为序号、算子名或接口名、shape大小错误值、shape大小正确值：

```text
The %sth input of %s has incorrect shape size %s. It should be %s.
```

报错示例如下：

```text
The 1th input of Conv2D has incorrect shape size 3. It should be 4.
```

## 解决方法

检查输入tensor的shape是否正确。
