# EZ0027 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、算子名或接口名、错误值、报错原因：

```text
Parameters %s of %s have incorrect values %s. Reason: %s.
```

报错示例如下：

```text
Parameters align_corners and half_pixel_centers of ResizeNearestNeighborV2Grad have incorrect values true and true. Reason: The values of attributes align_corners and half_pixel_centers cannot be true at the same time.
```

## 解决方法

根据报错原因检查参数值是否满足条件。
