# EZ1013 Config\_Error

## 错误信息

报错格式如下，占位符%s表示参数名：

```text
In the dynamic shape scenario, the JSON configuration file of operator %s cannot be found.
```

报错示例如下：

```text
In the dynamic shape scenario, the JSON configuration file of operator aclnnAdd_1_AddAiCore cannot be found.
```

## 可能原因

1. 未正确安装对应soc version的算子包。
2. 如果是自定义算子，可能是由于自定义算子环境变量ASCEND\_CUSTOM\_OPP\_PATH没有配置或者配置错误。
3. 如果是内置算子，可能是由于内置算子环境变量ASCEND\_OPP\_PATH没有配置或者配置错误。

## 解决方法

1. 请正确安装对应soc version的算子包。
2. 请设置ASCEND\_CUSTOM\_OPP\_PATH为自定算子包实际安装路径。
3. 请设置ASCEND\_OPP\_PATH为内置算子包实际安装路径。
