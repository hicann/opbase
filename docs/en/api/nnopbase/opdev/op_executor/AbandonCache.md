# AbandonCache

## Function

Disables the ACLNN cache.

After the first call to an ACLNN operator, the framework records cache information. On subsequent calls, it skips the initial `aclnnXxxGetWorkspaceSize` phase, thereby achieving performance optimization.

## Prototype

```cpp
void AbandonCache(bool disableRepeat = false)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| disableRepeat | Input| Whether to disable `aclOpExecutor` reuse. `true` marks the executor as unrepeatable. The default value, `false`, leaves its current repeat mode unchanged. Cache invalidation is performed for either value.|

## Returns

None

## Restrictions

If the `aclnnXxxGetWorkspaceSize` function must be executed in the initial phase, `AbandonCache` needs to be explicitly called in this function.

## Examples

```cpp
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    uniqueExecutor.get()->AbandonCache();
}
```
