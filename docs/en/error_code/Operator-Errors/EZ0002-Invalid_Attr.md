# EZ0002 Invalid\_Attr

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: attribute name, operator name or API name, incorrect value, correct value.

```text
Attribute %s of %s has incorrect value %s. It should be %s.
```

Error example:

```text
Attribute dim of KthValue has incorrect value 5. It should be [-3, 2].
```

## Solution

Check whether the operator attribute is correct.
