# EZ1013 Config\_Error

## Symptom

The following is error format. The placeholder %s indicates the operator name.

```text
In the dynamic shape scenario, the JSON configuration file of operator %s cannot be found.
```

Error example:

```text
In the dynamic shape scenario, the JSON configuration file of operator aclnnAdd_1_AddAiCore cannot be found.
```

## Possible Cause

1.The operator package of the corresponding SoC version is not correctly installed.

2.If the operator is a custom operator, the environment variable ASCEND\_CUSTOM\_OPP\_PATH of the custom operator may not be configured or be incorrectly configured.

3.If the operator is a built-in operator, the environment variable ASCEND\_OPP\_PATH of the built-in operator may not be configured or be incorrectly configured.

## Solution

1.Install the operator package of the corresponding SoC version correctly.

2.Set ASCEND\_CUSTOM\_OPP\_PATH to the actual installation path of the custom operator package.

3.Set ASCEND\_OPP\_PATH to the actual installation path of the built-in operator package.
