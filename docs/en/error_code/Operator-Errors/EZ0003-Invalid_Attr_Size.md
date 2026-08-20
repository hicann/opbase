# EZ0003 Invalid\_Attr\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: attribute name, operator name or API name, incorrect size, correct size.

```text
Attribute %s of %s has incorrect size %s. It should be %s.
```

Error example:

```text
Attribute strides of ExtendConvTranspose has incorrect size 2. It should be 3. 
```

## Solution

Check whether the operator attribute is correct.
