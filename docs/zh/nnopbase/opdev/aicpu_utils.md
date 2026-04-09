# aicpu\_uitls<a name="ZH-CN_TOPIC_0000002515603051"></a>

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

<a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_table10181182215132"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_row1918116227138"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p11181822161320"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p11181822161320"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p11181822161320"></a>接口定义</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p5181112213137"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p5181112213137"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p5181112213137"></a>功能说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_row12181172271314"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p151889175299"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p151889175299"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p151889175299"></a>RecordAicpuTime(const size_t index)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1821618215430"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1821618215430"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1821618215430"></a>记录执行算子任务时当前时间戳。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_row2018110223134"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1818812175299"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1818812175299"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1818812175299"></a>lk(gTimeStampMutex)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p32151211438"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p32151211438"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p32151211438"></a>时间戳数据结构的锁。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_row9181192201317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1645918133611"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1645918133611"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p1645918133611"></a>clock_gettime(CLOCK_MONOTONIC, &amp;(gAicpuTimeStamp.tp[index]))</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p421572112435"><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p421572112435"></a><a name="zh-cn_topic_0000002114053113_zh-cn_topic_0000001919471342_p421572112435"></a>系统函数，获取当前系统时间戳。</p>
</td>
</tr>
</tbody>
</table>

