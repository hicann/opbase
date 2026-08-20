# EZ0036 Invalid\_Argument\_Graph

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: invalid node name, error cause.

```text
The input graph contains an invalid node %s. Reason: %s.
```

Error example:

```text
The input graph contains an invalid node bias_1. Reason: In the fusion pass Conv2DPostCubeToExtendConv2DFusionPass, the output node connected to this node can only be of the following types: AscendDequant, AscendRequant.
```

## Solution

1. Refer to the official document for details about the dump graph.

2. Confirm and modify the input graph structure.
