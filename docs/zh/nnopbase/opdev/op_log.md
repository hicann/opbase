# op\_log<a name="ZH-CN_TOPIC_0000002483403090"></a>

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

<a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_table10181182215132"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row1918116227138"><th class="cellrowborder" valign="top" width="52.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p11181822161320"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p11181822161320"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p11181822161320"></a>接口定义</p>
</th>
<th class="cellrowborder" valign="top" width="47.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p5181112213137"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p5181112213137"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p5181112213137"></a>功能说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row12181172271314"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1983913122620"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1983913122620"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1983913122620"></a>GetTid()</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p19398722112610"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p19398722112610"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p19398722112610"></a>获取线程id。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row2018110223134"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p148391812261"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p148391812261"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p148391812261"></a>CheckLogLevelInner(int32_t moduleId, int32_t logLevel)</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1039892212265"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1039892212265"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p1039892212265"></a>检查log等级。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row9181192201317"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p3839817267"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p3839817267"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p3839817267"></a>DlogRecordInner(int32_t moduleId, int32_t level, const char *fmt, ...)</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p123981221264"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p123981221264"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p123981221264"></a>打印log。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row01811122131319"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p983919172619"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p983919172619"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p983919172619"></a>ReportErrorMessageInner(const std::string &amp;code, const char *fmt, ...)</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p203981522192616"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p203981522192616"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p203981522192616"></a>报告内部错误消息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_row1825693013416"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_p925617306414"><a name="zh-cn_topic_0000002114053081_p925617306414"></a><a name="zh-cn_topic_0000002114053081_p925617306414"></a>ReportErrorMessage(const char *code, const char *fmt, Arguments &amp;&amp;... args)</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_p1694416331846"><a name="zh-cn_topic_0000002114053081_p1694416331846"></a><a name="zh-cn_topic_0000002114053081_p1694416331846"></a>报告错误消息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row8181192221312"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p58402102613"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p58402102613"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p58402102613"></a>GetLogApiInfo()</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p103981022152617"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p103981022152617"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p103981022152617"></a>获取api信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row7182522101317"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p13840131162620"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p13840131162620"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p13840131162620"></a>GetFileName(const char *path)</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p14399112232614"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p14399112232614"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p14399112232614"></a>获取文件名。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_row568365122612"><td class="cellrowborder" valign="top" width="52.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p6840110264"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p6840110264"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p6840110264"></a>GetOpName()</p>
</td>
<td class="cellrowborder" valign="top" width="47.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p173991122132612"><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p173991122132612"></a><a name="zh-cn_topic_0000002114053081_zh-cn_topic_0000001919311402_p173991122132612"></a>获取算子名称。</p>
</td>
</tr>
</tbody>
</table>

