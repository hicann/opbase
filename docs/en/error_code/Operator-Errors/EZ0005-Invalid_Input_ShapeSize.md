# EZ0005 Invalid\_Input\_ShapeSize

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: index, operator name or API name, incorrect shape size, correct shape size.

```text
The %sth input of %s has incorrect shape size %s. It should be %s.
```

Error example:

```text
The 1th input of Conv2D has incorrect shape size 3. It should be 4.
```

## Solution

Check whether the shape of the input tensor is correct.
