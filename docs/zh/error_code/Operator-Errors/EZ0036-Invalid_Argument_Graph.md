# EZ0036 Invalid\_Argument\_Graph

## 错误信息

报错格式如下，占位符%s的含义依次为节点名、报错原因：

```text
The input graph contains an invalid node %s. Reason: %s.
```

报错示例如下：

```text
The input graph contains an invalid node bias_1. Reason: In the fusion pass Conv2DPostCubeToExtendConv2DFusionPass, the output node connected to this node can only be of the following types: AscendDequant, AscendRequant.
```

## 解决方法

1. 有关dump图的详细信息，请参阅官方文档。
2. 请确认并修改输入图结构。
