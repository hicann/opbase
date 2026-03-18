# op\_cache<a name="ZH-CN_TOPIC_0000002483403080"></a>

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

<a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_table10181182215132"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1918116227138"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11181822161320"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11181822161320"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11181822161320"></a>接口定义</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5181112213137"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5181112213137"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5181112213137"></a>功能说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12181172271314"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14521235113317"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14521235113317"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14521235113317"></a>GetCacheBuf()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4901154519333"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4901154519333"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4901154519333"></a>获取aclnn cache保存需要的内存buf。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row2018110223134"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10452103553311"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10452103553311"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10452103553311"></a>CheckCacheable()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14901145163319"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14901145163319"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14901145163319"></a>判断当前host侧aclnn API是否支持aclnn cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row9181192201317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453203513336"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453203513336"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453203513336"></a>AddSeperator()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p090154519336"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p090154519336"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p090154519336"></a>在aclnn cache的hashkey计算依据中，添加一个分隔符。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row01811122131319"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p19453113520331"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p19453113520331"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p19453113520331"></a>AddParamToBuf(const T *addr, uint64_t size)</p>
</td>
<td class="cellrowborder" rowspan="18" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p166813434011"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p166813434011"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p166813434011"></a>在aclnn cache的hashkey计算依据中，添加指定数据结构的关键信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row161811622181311"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164532351336"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164532351336"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164532351336"></a>AddParamToBuf(const aclTensor *tensor)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row16461955173210"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p12453173513310"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p12453173513310"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p12453173513310"></a>AddParamToBuf(aclTensor *tensor)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row3646105523219"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p445353519338"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p445353519338"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p445353519338"></a>AddParamToBuf(const aclScalar *scalar)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row06462557320"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11453835123317"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11453835123317"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p11453835123317"></a>AddParamToBuf(const aclIntArray *value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1064775514327"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1645333513318"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1645333513318"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1645333513318"></a>AddParamToBuf(const aclBoolArray *value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12647755203218"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6453835133313"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6453835133313"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6453835133313"></a>AddParamToBuf(const aclFloatArray *value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row848405814329"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4453133518330"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4453133518330"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p4453133518330"></a>AddParamToBuf(const aclFp16Array *value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row348415588329"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16453113510330"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16453113510330"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16453113510330"></a>AddParamToBuf(const aclTensorList *tensors)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1448445813214"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345373553317"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345373553317"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345373553317"></a>AddParamToBuf(const aclScalarList *scalars)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row74852589323"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531335113313"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531335113313"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531335113313"></a>AddParamToBuf(aclTensorList *tensors)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row17485458183218"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453103518334"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453103518334"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17453103518334"></a>AddParamToBuf(aclScalarList *scalars)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1248519585328"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p174531635193320"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p174531635193320"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p174531635193320"></a>AddParamToBuf(const std::string &amp;s)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row9485358193215"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p945313350332"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p945313350332"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p945313350332"></a>AddParamToBuf(const aclDataType dtype)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row9427161033313"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531735153319"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531735153319"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164531735153319"></a>AddParamToBuf(const char *c)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row642831012339"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1845353573316"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1845353573316"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1845353573316"></a>AddParamToBuf(char *c)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row179882033318"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1245353510339"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1245353510339"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1245353510339"></a>AddParamToBuf()</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1998142019336"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p84531835123316"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p84531835123316"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p84531835123316"></a>AddParamToBuf(const T &amp;value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row15430103005914"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p75341049195913"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p75341049195913"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p75341049195913"></a>AddOpConfigInfoToBuf()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14304302594"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14304302594"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p14304302594"></a>在aclnn cache的hashkey计算依据中，添加算子的配置信息，如：确定性计算。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row209822012330"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p134541335153312"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p134541335153312"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p134541335153312"></a>InitExecutorCacheThreadLocal()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6992214341"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6992214341"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6992214341"></a>初始化aclnn cache相关的线程变量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row18981520153310"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745403513311"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745403513311"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745403513311"></a>GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1493224344"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1493224344"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1493224344"></a>尝试获取匹配上的aclnn cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row5980202339"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345416355335"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345416355335"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p345416355335"></a>CalculateHashKey(const std::tuple&lt;Args...&gt; &amp;t)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1791422203416"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1791422203416"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1791422203416"></a>根据输入输出计算aclnn cache的hashkey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row29972010336"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p154540353335"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p154540353335"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p154540353335"></a>GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize, const char *api, const INPUT_TUPLE &amp;in, const OUTPUT_TUPLE &amp;out)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p189522163419"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p189522163419"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p189522163419"></a>尝试获取与指定输入输出匹配的aclnn cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row79992016338"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164541135163315"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164541135163315"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p164541135163315"></a>OpCacheKey()</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p23401226163414"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p23401226163414"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p23401226163414"></a>OpCacheKey为aclnn cache的key数据结构，该函数为其构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1399172083317"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1945416356331"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1945416356331"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1945416356331"></a>OpCacheKey(uint8_t *buf_, size_t len_)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row899122014339"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8454183573316"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8454183573316"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8454183573316"></a>OpCacheKey(const OpCacheKey &amp;key)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1699120193319"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745493511338"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745493511338"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1745493511338"></a>ToString()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p73342369345"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p73342369345"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p73342369345"></a>OpCacheKey的打印函数，将内容表示为字符串形式。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row19919204335"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p117044873512"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p117044873512"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p117044873512"></a>operator()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1333413361347"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1333413361347"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1333413361347"></a>OpCacheKey的判等函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row099132014331"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p187047818353"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p187047818353"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p187047818353"></a>SetOpCacheKey(OpCacheKey &amp;key)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p833811289351"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p833811289351"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p833811289351"></a>设置当前输入/输出对应的OpCacheKey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row11991020103316"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27048853513"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27048853513"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27048853513"></a>AddrRule()</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p35521213193618"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p35521213193618"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p35521213193618"></a>描述了aclnn cache中记录的aclTensor与host侧API外部入参间的对应关系，用于aclnn cache的地址刷新。该函数为其构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row9100162033316"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170410811357"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170410811357"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170410811357"></a>AddrRule(bool w, uint64_t offset, int inx, int64_t l2Offset)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1610052013333"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470458193516"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470458193516"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470458193516"></a>OpExecCache()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5154110123617"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5154110123617"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p5154110123617"></a>OpExecCache描述了一份aclnn cache信息，该函数为其构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row410015204339"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13704168133514"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13704168133514"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13704168133514"></a>InitOpCacheKey()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715491083615"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715491083615"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715491083615"></a>初始化aclnn cache的hashkey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row186012173335"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1970418814358"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1970418814358"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1970418814358"></a>InitCacheData()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8154191015362"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8154191015362"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8154191015362"></a>初始化aclnn cache需要的外部信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row560151719336"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1770415815357"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1770415815357"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1770415815357"></a>RecordAddrRule(const aclTensor *t, AddrRule &amp;rule)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3154910183614"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3154910183614"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3154910183614"></a>在生成aclnn cache过程中，生成一个记录aclTensor与host侧API外部入参间的对应关系的对象。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row116015175335"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1370415813353"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1370415813353"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1370415813353"></a>AddLaunchTensor(const aclTensor *t, size_t dataLen)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p151546105366"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p151546105366"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p151546105366"></a>在生成aclnn cache过程中，针对某个aclTensor，生成cache记录。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row5601617163311"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p67041782350"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p67041782350"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p67041782350"></a>AddLaunchData(size_t dataLen)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17154151093619"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17154151093619"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17154151093619"></a>在aclnn cache所在的内存上，申请一块指定大小的空间，用于填写cache内容。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row360111743312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p147046810350"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p147046810350"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p147046810350"></a>SetCacheBuf(void *buf)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p81541310133610"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p81541310133610"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p81541310133610"></a>设置aclnn cache所用的临时内存。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row661121714334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p770411833512"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p770411833512"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p770411833512"></a>UpdateTensorAddr(void *workspaceAddr, const std::vector&lt;void *&gt; &amp;tensors)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p171541410163619"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p171541410163619"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p171541410163619"></a>在使用aclnn cache时，首先将当前host侧API的实际输入地址刷新到cache中。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row261151793318"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"></a>SetBlockDim(uint32_t blockDim)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"></a>在生成aclnn cache的过程中，记录cache对象使用的核数。(即将废弃，请使用SetNumBlocks(uint32_t numBlocks))</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row261151793318"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p170517823516"></a>SetNumBlocks(uint32_t numBlocks)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p815571093617"></a>在生成aclnn cache的过程中，记录cache对象使用的核数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row2061161719334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27053893514"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27053893514"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p27053893514"></a>SetCacheTensorInfo(void *infoLists)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p21551910113618"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p21551910113618"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p21551910113618"></a>为了使aclnn cache支持profiling上报，收集aclTensor的属性信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row16616172336"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3705118103510"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3705118103510"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p3705118103510"></a>GetCacheTensorInfo(int index)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1915521014367"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1915521014367"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1915521014367"></a>为了使aclnn cache支持profiling上报，获取aclTensor的属性信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1661121793317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p970513833514"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p970513833514"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p970513833514"></a>AddTensorRelation(const aclTensor *tensorOut, const aclTensor *tensorMiddle)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p41555106361"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p41555106361"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p41555106361"></a>在aclnn cache中，记录两块aclTensor的内存等价关系。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12611817173317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470512883518"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470512883518"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p470512883518"></a>NewLaunchCache(size_t *offset, size_t *cap, R runner)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1315501033613"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1315501033613"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1315501033613"></a>生成一块新的cache内存。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row11615174334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p570558183514"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p570558183514"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p570558183514"></a>OldCacheClear()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715531073610"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715531073610"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p715531073610"></a>清理被淘汰的cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row196113178334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17051816353"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17051816353"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17051816353"></a>Finalize()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p141551510173617"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p141551510173617"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p141551510173617"></a>完成aclnn cache的生成工作，将可用标记位置位。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row18624173332"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p185428123715"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p185428123715"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p185428123715"></a>SetWorkspaceSize(uint64_t val)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p977992503719"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p977992503719"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p977992503719"></a>在生成aclnn cache的过程中，记录cache对象使用的workspace大小。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row146291793312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1354584376"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1354584376"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1354584376"></a>GetWorkspaceSize()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p0779172543712"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p0779172543712"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p0779172543712"></a>在使用aclnn cache时，获取cache对象使用的workspace大小。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row9621217193318"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165410843717"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165410843717"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165410843717"></a>GetHash()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p677902563716"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p677902563716"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p677902563716"></a>获取当前cache的hashkey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row17627176333"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p205428183720"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p205428183720"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p205428183720"></a>GetOpCacheKey()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p877982503714"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p877982503714"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p877982503714"></a>获取当前cache的hashkey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row26214175332"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2549810378"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2549810378"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2549810378"></a>MarkOpCacheInvalid()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1178072583717"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1178072583717"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1178072583717"></a>将当前aclnn cache置为弃用状态。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row4428191014335"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p954148153710"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p954148153710"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p954148153710"></a>IsOpCacheValid()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2780112513372"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2780112513372"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p2780112513372"></a>查询当前aclnn cache是否为弃用状态。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1989751313332"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8542086373"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8542086373"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p8542086373"></a>DoSummaryProfiling(int index)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778002510373"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778002510373"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778002510373"></a>在使用aclnn cache时，进行profiling上报。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1389710133332"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541383373"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541383373"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541383373"></a>RestoreThreadLocal(int index)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1378052583717"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1378052583717"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1378052583717"></a>在使用aclnn cache时，重置当前的线程上下文。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12897913173316"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1554183375"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1554183375"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1554183375"></a>Run(void *workspaceAddr, const aclrtStream stream, const std::vector&lt;void *&gt; &amp;tensors)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10780925183715"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10780925183715"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p10780925183715"></a>通过aclnn cache运行host侧API。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12897101383310"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165415863720"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165415863720"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p165415863720"></a>CanUse()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778082573710"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778082573710"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p778082573710"></a>设置aclnn cache的可用状态。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row168971613133313"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p65411819378"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p65411819378"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p65411819378"></a>SetUse()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p478042523712"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p478042523712"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p478042523712"></a>获取aclnn cache的可用状态。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row78976135330"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6542883718"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6542883718"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p6542883718"></a>GetShrinkList()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13780132513372"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13780132513372"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13780132513372"></a>获取当前的aclnn cache淘汰列表。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row589713131336"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p195420813712"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p195420813712"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p195420813712"></a>GetStorageRelation()</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p197801225173714"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p197801225173714"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p197801225173714"></a>获取aclnn cache中记录的aclTensor内存等价关系表。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row18897513143310"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p85412833711"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p85412833711"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p85412833711"></a>OpCacheValue()</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p188421132103717"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p188421132103717"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p188421132103717"></a>OpCacheValue是aclnn cache在LRU（Least Recently Used）中的数据节点封装，该函数为其构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row138981134337"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17541188371"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17541188371"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p17541188371"></a>OpCacheValue(OpExecCache *cache, OpCacheKey &amp;key)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1989841373310"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p654128123716"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p654128123716"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p654128123716"></a>OpCacheValue(const OpCacheValue &amp;value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1489811133334"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541819377"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541819377"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p13541819377"></a>OpCacheValue(OpCacheValue &amp;&amp;value)</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row12898131310334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p125448113715"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p125448113715"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p125448113715"></a>GetOpExecCache(uint64_t hash)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16727048173712"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16727048173712"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p16727048173712"></a>通过hashkey获取一个aclnn cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row1898141353313"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p145414816374"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p145414816374"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p145414816374"></a>GetOpExecCache(OpCacheKey &amp;key)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1072704817374"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1072704817374"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1072704817374"></a>通过hashkey获取一个aclnn cache。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row13898121353315"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1355682374"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1355682374"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1355682374"></a>AddOpExecCache(OpExecCache *exec)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p107271848183717"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p107271848183717"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p107271848183717"></a>向aclnn cache全局管理中，添加一个aclnn cache对象。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_row8898201311334"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1555158103716"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1555158103716"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1555158103716"></a>RemoveExecCache(OpExecCache *exec)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1272784816375"><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1272784816375"></a><a name="zh-cn_topic_0000002078498990_zh-cn_topic_0000001919311394_p1272784816375"></a>在aclnn cache全局管理中，移除一个aclnn cache对象。</p>
</td>
</tr>
</tbody>
</table>
