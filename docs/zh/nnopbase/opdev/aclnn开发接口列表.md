# aclnn开发接口列表<a name="ZH-CN_TOPIC_0000002515682985"></a>

无论是CANN内置算子或自定义算子，均可通过aclnn API直调算子，无需提供IR（Intermediate Representation）定义。

本章提供了**实现aclnn API**所需的**框架基础能力接口**，比如算子执行器（opExecutor）处理、数据类型/格式/shape等，具体参见下表罗列的接口、常用宏和常用类等。

**头文件说明**：调用本章接口时请按实际情况include依赖的头文件，头文件路径为\$\{INSTALL\_DIR\}/include目录。其中\$\{INSTALL\_DIR\}请替换为CANN软件安装后的文件存储路径。若安装Ascend-cann-toolkit软件包，以root安装举例，安装后文件存储路径为`/usr/local/Ascend/cann`。

**表 1**  框架能力接口列表

<a name="zh-cn_topic_0000002114052989_table2070319561334"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114052989_row117031456131"><th class="cellrowborder" valign="top" width="28.52%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002114052989_p4703125618315"><a name="zh-cn_topic_0000002114052989_p4703125618315"></a><a name="zh-cn_topic_0000002114052989_p4703125618315"></a>接口分类</p>
</th>
<th class="cellrowborder" valign="top" width="46.61%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002114052989_p177031056031"><a name="zh-cn_topic_0000002114052989_p177031056031"></a><a name="zh-cn_topic_0000002114052989_p177031056031"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="24.87%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002114052989_p3703105616316"><a name="zh-cn_topic_0000002114052989_p3703105616316"></a><a name="zh-cn_topic_0000002114052989_p3703105616316"></a>所属头文件</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114052989_row16703756535"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p470335617318"><a name="zh-cn_topic_0000002114052989_p470335617318"></a><a name="zh-cn_topic_0000002114052989_p470335617318"></a><a href="bfloat16.md">bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p270385618310"><a name="zh-cn_topic_0000002114052989_p270385618310"></a><a name="zh-cn_topic_0000002114052989_p270385618310"></a>详细介绍了bfloat16数据类型在CPU侧的实现类。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1570315617310"><a name="zh-cn_topic_0000002114052989_p1570315617310"></a><a name="zh-cn_topic_0000002114052989_p1570315617310"></a>aclnn/opdev/bfloat16.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1870355612313"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1170310569313"><a name="zh-cn_topic_0000002114052989_p1170310569313"></a><a name="zh-cn_topic_0000002114052989_p1170310569313"></a><a href="common_types.md">common_types</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1370314562037"><a name="zh-cn_topic_0000002114052989_p1370314562037"></a><a name="zh-cn_topic_0000002114052989_p1370314562037"></a>详细介绍了aclTensor、aclScalar等基础的aclnn数据结构。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p87031156937"><a name="zh-cn_topic_0000002114052989_p87031156937"></a><a name="zh-cn_topic_0000002114052989_p87031156937"></a>aclnn/opdev/common_types.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row7703145618314"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p47030569319"><a name="zh-cn_topic_0000002114052989_p47030569319"></a><a name="zh-cn_topic_0000002114052989_p47030569319"></a><a href="data_type_utils.md">data_type_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p17703105618320"><a name="zh-cn_topic_0000002114052989_p17703105618320"></a><a name="zh-cn_topic_0000002114052989_p17703105618320"></a>提供了DataType相关的基础接口，例如获取指定DataType的size等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p67031356735"><a name="zh-cn_topic_0000002114052989_p67031356735"></a><a name="zh-cn_topic_0000002114052989_p67031356735"></a>aclnn/opdev/data_type_utils.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row37036561318"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p207043565310"><a name="zh-cn_topic_0000002114052989_p207043565310"></a><a name="zh-cn_topic_0000002114052989_p207043565310"></a>预留接口，开发者无需关注。</p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p670465618318"><a name="zh-cn_topic_0000002114052989_p670465618318"></a><a name="zh-cn_topic_0000002114052989_p670465618318"></a>详细介绍了FastVector数据类型，该类型为aclnn中实现的高效vector数据结构。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p18704205615320"><a name="zh-cn_topic_0000002114052989_p18704205615320"></a><a name="zh-cn_topic_0000002114052989_p18704205615320"></a>aclnn/opdev/fast_vector.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row187047565318"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p87041956434"><a name="zh-cn_topic_0000002114052989_p87041956434"></a><a name="zh-cn_topic_0000002114052989_p87041956434"></a><a href="format_utils.md">format_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p3704856433"><a name="zh-cn_topic_0000002114052989_p3704856433"></a><a name="zh-cn_topic_0000002114052989_p3704856433"></a>提供了Format相关的基础接口。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p97043561035"><a name="zh-cn_topic_0000002114052989_p97043561035"></a><a name="zh-cn_topic_0000002114052989_p97043561035"></a>aclnn/opdev/format_utils.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row19704956437"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1770416561234"><a name="zh-cn_topic_0000002114052989_p1770416561234"></a><a name="zh-cn_topic_0000002114052989_p1770416561234"></a><a href="fp16_t.md">fp16_t</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1770417560316"><a name="zh-cn_topic_0000002114052989_p1770417560316"></a><a name="zh-cn_topic_0000002114052989_p1770417560316"></a>详细介绍了float16数据类型在CPU侧的实现类。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p27046561135"><a name="zh-cn_topic_0000002114052989_p27046561135"></a><a name="zh-cn_topic_0000002114052989_p27046561135"></a>aclnn/opdev/fp16_t.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row147043563317"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p47041956634"><a name="zh-cn_topic_0000002114052989_p47041956634"></a><a name="zh-cn_topic_0000002114052989_p47041956634"></a><a href="framework_op.md">framework_op</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1270412561532"><a name="zh-cn_topic_0000002114052989_p1270412561532"></a><a name="zh-cn_topic_0000002114052989_p1270412561532"></a>详细介绍了框架对外提供的从host侧到device侧拷贝能力。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p17041156239"><a name="zh-cn_topic_0000002114052989_p17041156239"></a><a name="zh-cn_topic_0000002114052989_p17041156239"></a>aclnn/opdev/framework_op.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row177049561320"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1470413561832"><a name="zh-cn_topic_0000002114052989_p1470413561832"></a><a name="zh-cn_topic_0000002114052989_p1470413561832"></a>预留接口，开发者无需关注。</p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p19704656232"><a name="zh-cn_topic_0000002114052989_p19704656232"></a><a name="zh-cn_topic_0000002114052989_p19704656232"></a>对外提供初始化aclOpExecutor的宏声明。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p14704256334"><a name="zh-cn_topic_0000002114052989_p14704256334"></a><a name="zh-cn_topic_0000002114052989_p14704256334"></a>aclnn/opdev/make_op_executor.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1970495620313"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p16704135612315"><a name="zh-cn_topic_0000002114052989_p16704135612315"></a><a name="zh-cn_topic_0000002114052989_p16704135612315"></a><a href="object.md">object</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p2070411564318"><a name="zh-cn_topic_0000002114052989_p2070411564318"></a><a name="zh-cn_topic_0000002114052989_p2070411564318"></a>详细介绍了aclnn中aclTensor等基础数据结构的基类Object类，用于重载实现new、delete方法。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p170420560315"><a name="zh-cn_topic_0000002114052989_p170420560315"></a><a name="zh-cn_topic_0000002114052989_p170420560315"></a>aclnn/opdev/object.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row270418562316"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p9704656336"><a name="zh-cn_topic_0000002114052989_p9704656336"></a><a name="zh-cn_topic_0000002114052989_p9704656336"></a><a href="op_arg_def.md">op_arg_def</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p07041456630"><a name="zh-cn_topic_0000002114052989_p07041456630"></a><a name="zh-cn_topic_0000002114052989_p07041456630"></a>详细介绍了OpArgContext类，并对外提供OP_INPUT等宏声明。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p07044560310"><a name="zh-cn_topic_0000002114052989_p07044560310"></a><a name="zh-cn_topic_0000002114052989_p07044560310"></a>aclnn/opdev/op_arg_def.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row07041056735"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p107051156932"><a name="zh-cn_topic_0000002114052989_p107051156932"></a><a name="zh-cn_topic_0000002114052989_p107051156932"></a><a href="op_cache.md">op_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p10705155620320"><a name="zh-cn_topic_0000002114052989_p10705155620320"></a><a name="zh-cn_topic_0000002114052989_p10705155620320"></a>详细介绍了OpExecCache及相关类，用于完成aclnn缓存，提升运行性能。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p2705175612319"><a name="zh-cn_topic_0000002114052989_p2705175612319"></a><a name="zh-cn_topic_0000002114052989_p2705175612319"></a>aclnn/opdev/op_cache.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row370517561733"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p2705956930"><a name="zh-cn_topic_0000002114052989_p2705956930"></a><a name="zh-cn_topic_0000002114052989_p2705956930"></a><a href="op_cache_container.md">op_cache_container</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p14705155618310"><a name="zh-cn_topic_0000002114052989_p14705155618310"></a><a name="zh-cn_topic_0000002114052989_p14705155618310"></a>详细介绍了带LRU淘汰机制的aclnn缓存容器。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p187053561937"><a name="zh-cn_topic_0000002114052989_p187053561937"></a><a name="zh-cn_topic_0000002114052989_p187053561937"></a>aclnn/opdev/op_cache_container.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1270517561235"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p2070515568317"><a name="zh-cn_topic_0000002114052989_p2070515568317"></a><a name="zh-cn_topic_0000002114052989_p2070515568317"></a><a href="op_config.md">op_config</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p0705125618313"><a name="zh-cn_topic_0000002114052989_p0705125618313"></a><a name="zh-cn_topic_0000002114052989_p0705125618313"></a>提供了算子运行时相关的配置信息，如确定性计算开关等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p270565611317"><a name="zh-cn_topic_0000002114052989_p270565611317"></a><a name="zh-cn_topic_0000002114052989_p270565611317"></a>aclnn/opdev/op_config.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row187052561436"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p67052561737"><a name="zh-cn_topic_0000002114052989_p67052561737"></a><a name="zh-cn_topic_0000002114052989_p67052561737"></a><a href="op_def.md">op_def</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p157059569316"><a name="zh-cn_topic_0000002114052989_p157059569316"></a><a name="zh-cn_topic_0000002114052989_p157059569316"></a>定义基础枚举及常量，例如精度模式OpImplMode等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p97051856336"><a name="zh-cn_topic_0000002114052989_p97051856336"></a><a name="zh-cn_topic_0000002114052989_p97051856336"></a>aclnn/opdev/op_def.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row670535613317"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1170511563312"><a name="zh-cn_topic_0000002114052989_p1170511563312"></a><a name="zh-cn_topic_0000002114052989_p1170511563312"></a><a href="op_dfx.md">op_dfx</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p17051356032"><a name="zh-cn_topic_0000002114052989_p17051356032"></a><a name="zh-cn_topic_0000002114052989_p17051356032"></a>详细介绍了DfxGuard类，用于接口打印及上报profiling。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p270512561312"><a name="zh-cn_topic_0000002114052989_p270512561312"></a><a name="zh-cn_topic_0000002114052989_p270512561312"></a>aclnn/opdev/op_dfx.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row770515617312"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><a href="aclnn返回码.md">aclnn返回码</a></td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p137051563316"><a name="zh-cn_topic_0000002114052989_p137051563316"></a><a name="zh-cn_topic_0000002114052989_p137051563316"></a>定义了aclnn错误码。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p27053561133"><a name="zh-cn_topic_0000002114052989_p27053561133"></a><a name="zh-cn_topic_0000002114052989_p27053561133"></a>aclnn/opdev/op_errno.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row370519561336"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p12706135611319"><a name="zh-cn_topic_0000002114052989_p12706135611319"></a><a name="zh-cn_topic_0000002114052989_p12706135611319"></a><a href="op_executor.md">op_executor</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p970665617311"><a name="zh-cn_topic_0000002114052989_p970665617311"></a><a name="zh-cn_topic_0000002114052989_p970665617311"></a>详细介绍了aclOpExecutor类。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1870612561433"><a name="zh-cn_topic_0000002114052989_p1870612561433"></a><a name="zh-cn_topic_0000002114052989_p1870612561433"></a>aclnn/opdev/op_executor.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row6706256035"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p5706125619318"><a name="zh-cn_topic_0000002114052989_p5706125619318"></a><a name="zh-cn_topic_0000002114052989_p5706125619318"></a><a href="op_log.md">op_log</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p470685616316"><a name="zh-cn_topic_0000002114052989_p470685616316"></a><a name="zh-cn_topic_0000002114052989_p470685616316"></a>定义aclnn中日志打印宏。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p370695620320"><a name="zh-cn_topic_0000002114052989_p370695620320"></a><a name="zh-cn_topic_0000002114052989_p370695620320"></a>aclnn/opdev/op_log.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1870655619315"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1570695613316"><a name="zh-cn_topic_0000002114052989_p1570695613316"></a><a name="zh-cn_topic_0000002114052989_p1570695613316"></a><a href="platform.md">platform</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p147061256139"><a name="zh-cn_topic_0000002114052989_p147061256139"></a><a name="zh-cn_topic_0000002114052989_p147061256139"></a>详细介绍了PlatformInfo类，用于存放SOC平台信息。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1470655617310"><a name="zh-cn_topic_0000002114052989_p1470655617310"></a><a name="zh-cn_topic_0000002114052989_p1470655617310"></a>aclnn/opdev/platform.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row87064561532"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p070619567311"><a name="zh-cn_topic_0000002114052989_p070619567311"></a><a name="zh-cn_topic_0000002114052989_p070619567311"></a><a href="pool_allocator.md">pool_allocator</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1770617562320"><a name="zh-cn_topic_0000002114052989_p1770617562320"></a><a name="zh-cn_topic_0000002114052989_p1770617562320"></a>详细介绍了PoolAllocator类，用于实现aclnn内部的CPU内存池。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1070616561930"><a name="zh-cn_topic_0000002114052989_p1070616561930"></a><a name="zh-cn_topic_0000002114052989_p1070616561930"></a>aclnn/opdev/pool_allocator.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1870625614317"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p13706356731"><a name="zh-cn_topic_0000002114052989_p13706356731"></a><a name="zh-cn_topic_0000002114052989_p13706356731"></a><a href="shape_utils.md">shape_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p187061756632"><a name="zh-cn_topic_0000002114052989_p187061756632"></a><a name="zh-cn_topic_0000002114052989_p187061756632"></a>提供了shape相关的基础操作，例如shape打印等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p370611561439"><a name="zh-cn_topic_0000002114052989_p370611561439"></a><a name="zh-cn_topic_0000002114052989_p370611561439"></a>aclnn/opdev/shape_utils.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row270645612318"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p20707185615311"><a name="zh-cn_topic_0000002114052989_p20707185615311"></a><a name="zh-cn_topic_0000002114052989_p20707185615311"></a><a href="small_vector.md">small_vector</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p37075569313"><a name="zh-cn_topic_0000002114052989_p37075569313"></a><a name="zh-cn_topic_0000002114052989_p37075569313"></a>详细介绍了SmallVector类，该类为aclnn中实现的高效vector数据结构，主要针对已知数据量较小的场景。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p177071056139"><a name="zh-cn_topic_0000002114052989_p177071056139"></a><a name="zh-cn_topic_0000002114052989_p177071056139"></a>aclnn/opdev/small_vector.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row37072561731"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p87071561130"><a name="zh-cn_topic_0000002114052989_p87071561130"></a><a name="zh-cn_topic_0000002114052989_p87071561130"></a><a href="tensor_view_utils.md">tensor_view_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p770711561534"><a name="zh-cn_topic_0000002114052989_p770711561534"></a><a name="zh-cn_topic_0000002114052989_p770711561534"></a>提供了对于View类的基础操作，例如判断aclTensor是否连续等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1270795614317"><a name="zh-cn_topic_0000002114052989_p1270795614317"></a><a name="zh-cn_topic_0000002114052989_p1270795614317"></a>aclnn/opdev/tensor_view_utils.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row970719561439"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p3707756934"><a name="zh-cn_topic_0000002114052989_p3707756934"></a><a name="zh-cn_topic_0000002114052989_p3707756934"></a><a href="data_type_utils-6.md">data_type_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p270819561538"><a name="zh-cn_topic_0000002114052989_p270819561538"></a><a name="zh-cn_topic_0000002114052989_p270819561538"></a>提供了DataType相关的基础接口，例如判断指定DataType是否为整数类型等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p570810569315"><a name="zh-cn_topic_0000002114052989_p570810569315"></a><a name="zh-cn_topic_0000002114052989_p570810569315"></a>aclnn/opdev/op_common/data_type_utils.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row470865611316"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1370819561037"><a name="zh-cn_topic_0000002114052989_p1370819561037"></a><a name="zh-cn_topic_0000002114052989_p1370819561037"></a><a href="aicpu_args_handler.md">aicpu_args_handler</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p137084561235"><a name="zh-cn_topic_0000002114052989_p137084561235"></a><a name="zh-cn_topic_0000002114052989_p137084561235"></a>提供了AI CPU相关的组合计算任务的处理逻辑，例如拼接计算任务相关的参数等。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1670835610311"><a name="zh-cn_topic_0000002114052989_p1670835610311"></a><a name="zh-cn_topic_0000002114052989_p1670835610311"></a>aclnn/opdev/aicpu/aicpu_args_handler.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row147082569311"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p177081656934"><a name="zh-cn_topic_0000002114052989_p177081656934"></a><a name="zh-cn_topic_0000002114052989_p177081656934"></a><a href="aicpu_ext_info_handle.md">aicpu_ext_info_handle</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p147081756235"><a name="zh-cn_topic_0000002114052989_p147081756235"></a><a name="zh-cn_topic_0000002114052989_p147081756235"></a>提供了AI CPU相关的计算任务拓展参数的处理逻辑，例如拼接解析拓展参数的接口。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p197081256435"><a name="zh-cn_topic_0000002114052989_p197081256435"></a><a name="zh-cn_topic_0000002114052989_p197081256435"></a>aclnn/opdev/aicpu/aicpu_ext_info_handle.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row280425420348"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1070817569311"><a name="zh-cn_topic_0000002114052989_p1070817569311"></a><a name="zh-cn_topic_0000002114052989_p1070817569311"></a><a href="aicpu_task.md">aicpu_task</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p117089567312"><a name="zh-cn_topic_0000002114052989_p117089567312"></a><a name="zh-cn_topic_0000002114052989_p117089567312"></a>提供了AI CPU任务设置、下发等逻辑，例如设置调用哪个AI CPU算子，设置算子输入、输出等接口。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p11708195610316"><a name="zh-cn_topic_0000002114052989_p11708195610316"></a><a name="zh-cn_topic_0000002114052989_p11708195610316"></a>aclnn/opdev/aicpu/aicpu_task.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row19708175620310"><td class="cellrowborder" valign="top" width="28.52%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p127085561238"><a name="zh-cn_topic_0000002114052989_p127085561238"></a><a name="zh-cn_topic_0000002114052989_p127085561238"></a><a href="aicpu_utils.md">aicpu_utils</a></p>
</td>
<td class="cellrowborder" valign="top" width="46.61%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p970812561335"><a name="zh-cn_topic_0000002114052989_p970812561335"></a><a name="zh-cn_topic_0000002114052989_p970812561335"></a>AI CPU任务需要的一些公共接口。</p>
</td>
<td class="cellrowborder" valign="top" width="24.87%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p870825618320"><a name="zh-cn_topic_0000002114052989_p870825618320"></a><a name="zh-cn_topic_0000002114052989_p870825618320"></a>aclnn/opdev/aicpu/aicpu_utils.h</p>
</td>
</tr>
</tbody>
</table>

**表 2**  常用宏表

<a name="zh-cn_topic_0000002114052989_table370875616310"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114052989_row3708125617320"><th class="cellrowborder" valign="top" width="26.39%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002114052989_p67080561532"><a name="zh-cn_topic_0000002114052989_p67080561532"></a><a name="zh-cn_topic_0000002114052989_p67080561532"></a>宏名称</p>
</th>
<th class="cellrowborder" valign="top" width="48.559999999999995%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002114052989_p117091561332"><a name="zh-cn_topic_0000002114052989_p117091561332"></a><a name="zh-cn_topic_0000002114052989_p117091561332"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="25.05%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002114052989_p4709175615317"><a name="zh-cn_topic_0000002114052989_p4709175615317"></a><a name="zh-cn_topic_0000002114052989_p4709175615317"></a>所属头文件</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114052989_row170913566319"><td class="cellrowborder" valign="top" width="26.39%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p187091056934"><a name="zh-cn_topic_0000002114052989_p187091056934"></a><a name="zh-cn_topic_0000002114052989_p187091056934"></a><a href="DFX_IN.md">DFX_IN</a></p>
</td>
<td class="cellrowborder" valign="top" width="48.559999999999995%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p177091556431"><a name="zh-cn_topic_0000002114052989_p177091556431"></a><a name="zh-cn_topic_0000002114052989_p177091556431"></a>在L2_DFX_PHASE_1中，用于打包所有的host侧API输入参数。</p>
</td>
<td class="cellrowborder" rowspan="6" valign="top" width="25.05%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1170965610317"><a name="zh-cn_topic_0000002114052989_p1170965610317"></a><a name="zh-cn_topic_0000002114052989_p1170965610317"></a>aclnn/opdev/op_dfx.h</p>
<p id="zh-cn_topic_0000002114052989_p270925616311"><a name="zh-cn_topic_0000002114052989_p270925616311"></a><a name="zh-cn_topic_0000002114052989_p270925616311"></a></p>
<p id="zh-cn_topic_0000002114052989_p870917568317"><a name="zh-cn_topic_0000002114052989_p870917568317"></a><a name="zh-cn_topic_0000002114052989_p870917568317"></a></p>
<p id="zh-cn_topic_0000002114052989_p12709105616316"><a name="zh-cn_topic_0000002114052989_p12709105616316"></a><a name="zh-cn_topic_0000002114052989_p12709105616316"></a></p>
<p id="zh-cn_topic_0000002114052989_p13709256732"><a name="zh-cn_topic_0000002114052989_p13709256732"></a><a name="zh-cn_topic_0000002114052989_p13709256732"></a></p>
<p id="zh-cn_topic_0000002114052989_p1870985613313"><a name="zh-cn_topic_0000002114052989_p1870985613313"></a><a name="zh-cn_topic_0000002114052989_p1870985613313"></a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row19709156633"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1270910561132"><a name="zh-cn_topic_0000002114052989_p1270910561132"></a><a name="zh-cn_topic_0000002114052989_p1270910561132"></a><a href="DFX_OUT.md">DFX_OUT</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p137092561134"><a name="zh-cn_topic_0000002114052989_p137092561134"></a><a name="zh-cn_topic_0000002114052989_p137092561134"></a>在L2_DFX_PHASE_1中，用于打包所有的host侧API输出参数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row270935618313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p137091561638"><a name="zh-cn_topic_0000002114052989_p137091561638"></a><a name="zh-cn_topic_0000002114052989_p137091561638"></a><a href="L0_DFX.md">L0_DFX</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p107099563310"><a name="zh-cn_topic_0000002114052989_p107099563310"></a><a name="zh-cn_topic_0000002114052989_p107099563310"></a>必须在host侧API L0接口中使用，用于接口及L0接口入参打印。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row57098561034"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p470915618312"><a name="zh-cn_topic_0000002114052989_p470915618312"></a><a name="zh-cn_topic_0000002114052989_p470915618312"></a><a href="L2_DFX_PHASE_1.md">L2_DFX_PHASE_1</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1070916564312"><a name="zh-cn_topic_0000002114052989_p1070916564312"></a><a name="zh-cn_topic_0000002114052989_p1070916564312"></a>必须在一阶段接口最前方调用，用于接口及一阶段入参打印。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row15709205611311"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p177098561315"><a name="zh-cn_topic_0000002114052989_p177098561315"></a><a name="zh-cn_topic_0000002114052989_p177098561315"></a><a href="L2_DFX_PHASE_2.md">L2_DFX_PHASE_2</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p167096560319"><a name="zh-cn_topic_0000002114052989_p167096560319"></a><a name="zh-cn_topic_0000002114052989_p167096560319"></a>必须在二阶段接口最前方调用，用于接口打印。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row7709556830"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p37091856234"><a name="zh-cn_topic_0000002114052989_p37091856234"></a><a name="zh-cn_topic_0000002114052989_p37091856234"></a><a href="OP_TYPE_REGISTER.md">OP_TYPE_REGISTER</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p13709856539"><a name="zh-cn_topic_0000002114052989_p13709856539"></a><a name="zh-cn_topic_0000002114052989_p13709856539"></a>必须在L0接口最开始处使用，用于注册L0算子。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row137101567311"><td class="cellrowborder" valign="top" width="26.39%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p771055616313"><a name="zh-cn_topic_0000002114052989_p771055616313"></a><a name="zh-cn_topic_0000002114052989_p771055616313"></a><a href="OP_ATTR.md">OP_ATTR</a></p>
</td>
<td class="cellrowborder" valign="top" width="48.559999999999995%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1571020568311"><a name="zh-cn_topic_0000002114052989_p1571020568311"></a><a name="zh-cn_topic_0000002114052989_p1571020568311"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子属性参数。</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="25.05%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p771010564320"><a name="zh-cn_topic_0000002114052989_p771010564320"></a><a name="zh-cn_topic_0000002114052989_p771010564320"></a>aclnn/opdev/op_arg_def.h</p>
<p id="zh-cn_topic_0000002114052989_p471018561732"><a name="zh-cn_topic_0000002114052989_p471018561732"></a><a name="zh-cn_topic_0000002114052989_p471018561732"></a></p>
<p id="zh-cn_topic_0000002114052989_p12710195615310"><a name="zh-cn_topic_0000002114052989_p12710195615310"></a><a name="zh-cn_topic_0000002114052989_p12710195615310"></a></p>
<p id="zh-cn_topic_0000002114052989_p157106561230"><a name="zh-cn_topic_0000002114052989_p157106561230"></a><a name="zh-cn_topic_0000002114052989_p157106561230"></a></p>
<p id="zh-cn_topic_0000002114052989_p147101564318"><a name="zh-cn_topic_0000002114052989_p147101564318"></a><a name="zh-cn_topic_0000002114052989_p147101564318"></a></p>
<p id="zh-cn_topic_0000002114052989_p19710056332"><a name="zh-cn_topic_0000002114052989_p19710056332"></a><a name="zh-cn_topic_0000002114052989_p19710056332"></a></p>
<p id="zh-cn_topic_0000002114052989_p47103561532"><a name="zh-cn_topic_0000002114052989_p47103561532"></a><a name="zh-cn_topic_0000002114052989_p47103561532"></a></p>
<p id="zh-cn_topic_0000002114052989_p19710656231"><a name="zh-cn_topic_0000002114052989_p19710656231"></a><a name="zh-cn_topic_0000002114052989_p19710656231"></a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row157101456735"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p207108566311"><a name="zh-cn_topic_0000002114052989_p207108566311"></a><a name="zh-cn_topic_0000002114052989_p207108566311"></a><a href="OP_EMPTY_ARG.md">OP_EMPTY_ARG</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1710145612312"><a name="zh-cn_topic_0000002114052989_p1710145612312"></a><a name="zh-cn_topic_0000002114052989_p1710145612312"></a>ADD_TO_LAUNCHER_LIST_AICORE中，用于占位一个空的输入或输出。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row197101561333"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p77106569315"><a name="zh-cn_topic_0000002114052989_p77106569315"></a><a name="zh-cn_topic_0000002114052989_p77106569315"></a><a href="OP_INPUT.md">OP_INPUT</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p37109563314"><a name="zh-cn_topic_0000002114052989_p37109563314"></a><a name="zh-cn_topic_0000002114052989_p37109563314"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输入aclTensor。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row57101356635"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1071010561316"><a name="zh-cn_topic_0000002114052989_p1071010561316"></a><a name="zh-cn_topic_0000002114052989_p1071010561316"></a><a href="OP_MODE.md">OP_MODE</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p871095615319"><a name="zh-cn_topic_0000002114052989_p871095615319"></a><a name="zh-cn_topic_0000002114052989_p871095615319"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子运行选项，例如是否使能HF32。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1371019561936"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p137101156838"><a name="zh-cn_topic_0000002114052989_p137101156838"></a><a name="zh-cn_topic_0000002114052989_p137101156838"></a><a href="OP_OUTPUT.md">OP_OUTPUT</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p187102056836"><a name="zh-cn_topic_0000002114052989_p187102056836"></a><a name="zh-cn_topic_0000002114052989_p187102056836"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输出aclTensor。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row971016561739"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1871035612311"><a name="zh-cn_topic_0000002114052989_p1871035612311"></a><a name="zh-cn_topic_0000002114052989_p1871035612311"></a><a href="OP_OUTSHAPE.md">OP_OUTSHAPE</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p147113561313"><a name="zh-cn_topic_0000002114052989_p147113561313"></a><a name="zh-cn_topic_0000002114052989_p147113561313"></a>ADD_TO_LAUNCHER_LIST_AICORE中，针对第三类算子，设置存放输出shape的aclTensor。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row2071165616316"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1371118561037"><a name="zh-cn_topic_0000002114052989_p1371118561037"></a><a name="zh-cn_topic_0000002114052989_p1371118561037"></a><a href="OP_OPTION.md">OP_OPTION</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p27115561636"><a name="zh-cn_topic_0000002114052989_p27115561636"></a><a name="zh-cn_topic_0000002114052989_p27115561636"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子指定的精度模式。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row57118563313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1771135614320"><a name="zh-cn_topic_0000002114052989_p1771135614320"></a><a name="zh-cn_topic_0000002114052989_p1771135614320"></a><a href="OP_WORKSPACE.md">OP_WORKSPACE</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p771145619312"><a name="zh-cn_topic_0000002114052989_p771145619312"></a><a name="zh-cn_topic_0000002114052989_p771145619312"></a>ADD_TO_LAUNCHER_LIST_AICORE中，打包算子显式指定的workspace参数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row107110561332"><td class="cellrowborder" valign="top" width="26.39%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p2071145611314"><a name="zh-cn_topic_0000002114052989_p2071145611314"></a><a name="zh-cn_topic_0000002114052989_p2071145611314"></a><a href="CREATE_EXECUTOR.md">CREATE_EXECUTOR</a></p>
</td>
<td class="cellrowborder" valign="top" width="48.559999999999995%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p571119560310"><a name="zh-cn_topic_0000002114052989_p571119560310"></a><a name="zh-cn_topic_0000002114052989_p571119560310"></a>创建一个UniqueExecutor对象，该对象为aclOpExecutor的生成工厂类。</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="25.05%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p167118561231"><a name="zh-cn_topic_0000002114052989_p167118561231"></a><a name="zh-cn_topic_0000002114052989_p167118561231"></a>aclnn/opdev/make_op_executor.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row0711125613315"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p7711115617319"><a name="zh-cn_topic_0000002114052989_p7711115617319"></a><a name="zh-cn_topic_0000002114052989_p7711115617319"></a><a href="INFER_SHAPE.md">INFER_SHAPE</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p14711356138"><a name="zh-cn_topic_0000002114052989_p14711356138"></a><a name="zh-cn_topic_0000002114052989_p14711356138"></a>针对指定算子，运行其infershape函数，推导输出shape。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1671117563316"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p871119561038"><a name="zh-cn_topic_0000002114052989_p871119561038"></a><a name="zh-cn_topic_0000002114052989_p871119561038"></a><a href="ADD_TO_LAUNCHER_LIST_AICORE.md">ADD_TO_LAUNCHER_LIST_AICORE</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p77111556031"><a name="zh-cn_topic_0000002114052989_p77111556031"></a><a name="zh-cn_topic_0000002114052989_p77111556031"></a>创建某个AICore算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row67111056632"><td class="cellrowborder" valign="top" width="26.39%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1971110563313"><a name="zh-cn_topic_0000002114052989_p1971110563313"></a><a name="zh-cn_topic_0000002114052989_p1971110563313"></a><a href="OP_ATTR_NAMES.md">OP_ATTR_NAMES</a></p>
</td>
<td class="cellrowborder" valign="top" width="48.559999999999995%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p471118563312"><a name="zh-cn_topic_0000002114052989_p471118563312"></a><a name="zh-cn_topic_0000002114052989_p471118563312"></a>String类型的vector，打包AI CPU算子的字符类型属性。</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="25.05%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p37113562038"><a name="zh-cn_topic_0000002114052989_p37113562038"></a><a name="zh-cn_topic_0000002114052989_p37113562038"></a>aclnn/opdev/aicpu/aicpu_task.h</p>
<p id="zh-cn_topic_0000002114052989_p2071116561833"><a name="zh-cn_topic_0000002114052989_p2071116561833"></a><a name="zh-cn_topic_0000002114052989_p2071116561833"></a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row47111756538"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1671120563312"><a name="zh-cn_topic_0000002114052989_p1671120563312"></a><a name="zh-cn_topic_0000002114052989_p1671120563312"></a><a href="ADD_TO_LAUNCHER_LIST_AICPU.md">ADD_TO_LAUNCHER_LIST_AICPU</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1871215563320"><a name="zh-cn_topic_0000002114052989_p1871215563320"></a><a name="zh-cn_topic_0000002114052989_p1871215563320"></a>创建某个AI CPU算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  常用class和struct表

<a name="zh-cn_topic_0000002114052989_table2712056035"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002114052989_row137121561431"><th class="cellrowborder" valign="top" width="25.93259325932593%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002114052989_p12712185620318"><a name="zh-cn_topic_0000002114052989_p12712185620318"></a><a name="zh-cn_topic_0000002114052989_p12712185620318"></a>class/struct名称</p>
</th>
<th class="cellrowborder" valign="top" width="49.17491749174918%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002114052989_p27128561638"><a name="zh-cn_topic_0000002114052989_p27128561638"></a><a name="zh-cn_topic_0000002114052989_p27128561638"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="24.892489248924893%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002114052989_p57127563310"><a name="zh-cn_topic_0000002114052989_p57127563310"></a><a name="zh-cn_topic_0000002114052989_p57127563310"></a>所属头文件</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002114052989_row1071215616314"><td class="cellrowborder" valign="top" width="25.93259325932593%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p67120568311"><a name="zh-cn_topic_0000002114052989_p67120568311"></a><a name="zh-cn_topic_0000002114052989_p67120568311"></a>aclOpExecutor</p>
</td>
<td class="cellrowborder" valign="top" width="49.17491749174918%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p117129561531"><a name="zh-cn_topic_0000002114052989_p117129561531"></a><a name="zh-cn_topic_0000002114052989_p117129561531"></a>用于表示算子执行器，记录整个host侧API运行信息的上下文结构，如L2接口执行过程中的计算图、L0算子launch子任务、workspace地址和大小等信息。</p>
<p id="zh-cn_topic_0000002114052989_p1371219562038"><a name="zh-cn_topic_0000002114052989_p1371219562038"></a><a name="zh-cn_topic_0000002114052989_p1371219562038"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="op_executor.md">op_executor</a>。</p>
</td>
<td class="cellrowborder" valign="top" width="24.892489248924893%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1271212561231"><a name="zh-cn_topic_0000002114052989_p1271212561231"></a><a name="zh-cn_topic_0000002114052989_p1271212561231"></a>aclnn/opdev/op_executor.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row15712656535"><td class="cellrowborder" valign="top" width="25.93259325932593%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p1971212560319"><a name="zh-cn_topic_0000002114052989_p1971212560319"></a><a name="zh-cn_topic_0000002114052989_p1971212560319"></a>aclTensor</p>
</td>
<td class="cellrowborder" valign="top" width="49.17491749174918%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p4712356433"><a name="zh-cn_topic_0000002114052989_p4712356433"></a><a name="zh-cn_topic_0000002114052989_p4712356433"></a>用于表示一个张量对象，包括描述张量的shape、dtype、format、address等信息，数据可以放在host侧或device侧。</p>
<p id="zh-cn_topic_0000002114052989_p9712656232"><a name="zh-cn_topic_0000002114052989_p9712656232"></a><a name="zh-cn_topic_0000002114052989_p9712656232"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="24.892489248924893%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p167121356434"><a name="zh-cn_topic_0000002114052989_p167121356434"></a><a name="zh-cn_topic_0000002114052989_p167121356434"></a>aclnn/opdev/common_type.h</p>
<p id="zh-cn_topic_0000002114052989_p671225618320"><a name="zh-cn_topic_0000002114052989_p671225618320"></a><a name="zh-cn_topic_0000002114052989_p671225618320"></a></p>
<p id="zh-cn_topic_0000002114052989_p77127568319"><a name="zh-cn_topic_0000002114052989_p77127568319"></a><a name="zh-cn_topic_0000002114052989_p77127568319"></a></p>
<p id="zh-cn_topic_0000002114052989_p071212561039"><a name="zh-cn_topic_0000002114052989_p071212561039"></a><a name="zh-cn_topic_0000002114052989_p071212561039"></a></p>
<p id="zh-cn_topic_0000002114052989_p177126561532"><a name="zh-cn_topic_0000002114052989_p177126561532"></a><a name="zh-cn_topic_0000002114052989_p177126561532"></a></p>
<p id="zh-cn_topic_0000002114052989_p1071216566311"><a name="zh-cn_topic_0000002114052989_p1071216566311"></a><a name="zh-cn_topic_0000002114052989_p1071216566311"></a></p>
<p id="zh-cn_topic_0000002114052989_p571265614316"><a name="zh-cn_topic_0000002114052989_p571265614316"></a><a name="zh-cn_topic_0000002114052989_p571265614316"></a></p>
<p id="zh-cn_topic_0000002114052989_p97121561237"><a name="zh-cn_topic_0000002114052989_p97121561237"></a><a name="zh-cn_topic_0000002114052989_p97121561237"></a></p>
<p id="zh-cn_topic_0000002114052989_p1671211561931"><a name="zh-cn_topic_0000002114052989_p1671211561931"></a><a name="zh-cn_topic_0000002114052989_p1671211561931"></a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1071213561231"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p5712456437"><a name="zh-cn_topic_0000002114052989_p5712456437"></a><a name="zh-cn_topic_0000002114052989_p5712456437"></a>aclScalar</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p97132056733"><a name="zh-cn_topic_0000002114052989_p97132056733"></a><a name="zh-cn_topic_0000002114052989_p97132056733"></a>用于表示一个标量对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p157139569317"><a name="zh-cn_topic_0000002114052989_p157139569317"></a><a name="zh-cn_topic_0000002114052989_p157139569317"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row371316565320"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p87130564318"><a name="zh-cn_topic_0000002114052989_p87130564318"></a><a name="zh-cn_topic_0000002114052989_p87130564318"></a>aclTensorList</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1571310562315"><a name="zh-cn_topic_0000002114052989_p1571310562315"></a><a name="zh-cn_topic_0000002114052989_p1571310562315"></a>用于表示一组aclTensor类型组成的列表对象。</p>
<p id="zh-cn_topic_0000002114052989_p157136567311"><a name="zh-cn_topic_0000002114052989_p157136567311"></a><a name="zh-cn_topic_0000002114052989_p157136567311"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row197138561933"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p147131256035"><a name="zh-cn_topic_0000002114052989_p147131256035"></a><a name="zh-cn_topic_0000002114052989_p147131256035"></a>aclScalarList</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1571312567315"><a name="zh-cn_topic_0000002114052989_p1571312567315"></a><a name="zh-cn_topic_0000002114052989_p1571312567315"></a>用于表示一组aclScalar类型组成的列表对象。</p>
<p id="zh-cn_topic_0000002114052989_p97135561230"><a name="zh-cn_topic_0000002114052989_p97135561230"></a><a name="zh-cn_topic_0000002114052989_p97135561230"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row4713956638"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p187131456538"><a name="zh-cn_topic_0000002114052989_p187131456538"></a><a name="zh-cn_topic_0000002114052989_p187131456538"></a>aclBoolArray</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p13713205613314"><a name="zh-cn_topic_0000002114052989_p13713205613314"></a><a name="zh-cn_topic_0000002114052989_p13713205613314"></a>用于表示一个布尔类型的数组对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p57139569313"><a name="zh-cn_topic_0000002114052989_p57139569313"></a><a name="zh-cn_topic_0000002114052989_p57139569313"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row971305611310"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p57131956838"><a name="zh-cn_topic_0000002114052989_p57131956838"></a><a name="zh-cn_topic_0000002114052989_p57131956838"></a>aclIntArray</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1671315566311"><a name="zh-cn_topic_0000002114052989_p1671315566311"></a><a name="zh-cn_topic_0000002114052989_p1671315566311"></a>用于表示一个int64_t类型的数组对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p10713175617320"><a name="zh-cn_topic_0000002114052989_p10713175617320"></a><a name="zh-cn_topic_0000002114052989_p10713175617320"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row171312566319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p171311561133"><a name="zh-cn_topic_0000002114052989_p171311561133"></a><a name="zh-cn_topic_0000002114052989_p171311561133"></a>aclFloatArray</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p16713856533"><a name="zh-cn_topic_0000002114052989_p16713856533"></a><a name="zh-cn_topic_0000002114052989_p16713856533"></a>用于表示一个fp32类型的数组对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p1371315612312"><a name="zh-cn_topic_0000002114052989_p1371315612312"></a><a name="zh-cn_topic_0000002114052989_p1371315612312"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row20713185612316"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p471314561435"><a name="zh-cn_topic_0000002114052989_p471314561435"></a><a name="zh-cn_topic_0000002114052989_p471314561435"></a>aclFp16Array</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p10713656233"><a name="zh-cn_topic_0000002114052989_p10713656233"></a><a name="zh-cn_topic_0000002114052989_p10713656233"></a>用于表示一个fp16类型的数组对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p47131056137"><a name="zh-cn_topic_0000002114052989_p47131056137"></a><a name="zh-cn_topic_0000002114052989_p47131056137"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row67141556535"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p97148563314"><a name="zh-cn_topic_0000002114052989_p97148563314"></a><a name="zh-cn_topic_0000002114052989_p97148563314"></a>aclBf16Array</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p17141556435"><a name="zh-cn_topic_0000002114052989_p17141556435"></a><a name="zh-cn_topic_0000002114052989_p17141556435"></a>用于表示一个bf16类型的数组对象，数据一般放在host侧。</p>
<p id="zh-cn_topic_0000002114052989_p1071413565311"><a name="zh-cn_topic_0000002114052989_p1071413565311"></a><a name="zh-cn_topic_0000002114052989_p1071413565311"></a>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="common_types.md">common_types</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1714456932"><td class="cellrowborder" valign="top" width="25.93259325932593%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p10714105611319"><a name="zh-cn_topic_0000002114052989_p10714105611319"></a><a name="zh-cn_topic_0000002114052989_p10714105611319"></a>SmallVector</p>
</td>
<td class="cellrowborder" valign="top" width="49.17491749174918%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p77140561239"><a name="zh-cn_topic_0000002114052989_p77140561239"></a><a name="zh-cn_topic_0000002114052989_p77140561239"></a>该类使用内部内存池实现vector容器，基础功能与C++标准库中std::vector容器相同，无需每次扩容都申请内存，避免影响性能。其定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见<a href="small_vector.md">small_vector</a>。</p>
<a name="zh-cn_topic_0000002114052989_ul7714856639"></a><a name="zh-cn_topic_0000002114052989_ul7714856639"></a><ul id="zh-cn_topic_0000002114052989_ul7714856639"><li><strong id="zh-cn_topic_0000002114052989_b1662610281206"><a name="zh-cn_topic_0000002114052989_b1662610281206"></a><a name="zh-cn_topic_0000002114052989_b1662610281206"></a>op::FVector</strong>：本质是存储容量长度为8的SmallVector。<a name="zh-cn_topic_0000002114052989_screen0714155620317"></a><a name="zh-cn_topic_0000002114052989_screen0714155620317"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000002114052989_screen0714155620317">namespace op {
template&lt;typename T, size_t N = 8&gt;
using FVector = op::internal::SmallVector&lt;T, N, op::internal::PoolAllocator&lt;T&gt;&gt;;
}</pre>
</li><li><strong id="zh-cn_topic_0000002114052989_b129211318206"><a name="zh-cn_topic_0000002114052989_b129211318206"></a><a name="zh-cn_topic_0000002114052989_b129211318206"></a>op:Strides</strong>：本质是存储容量长度为25的FVector，元素类型int64_t，存储stride信息。<a name="zh-cn_topic_0000002114052989_screen15714556932"></a><a name="zh-cn_topic_0000002114052989_screen15714556932"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000002114052989_screen15714556932">namespace op {
constexpr uint64_t MAX_DIM_NUM = 25;
using Strides = FVector&lt;int64_t, MAX_DIM_NUM&gt;;
}</pre>
</li><li><strong id="zh-cn_topic_0000002114052989_b1729113364204"><a name="zh-cn_topic_0000002114052989_b1729113364204"></a><a name="zh-cn_topic_0000002114052989_b1729113364204"></a>op::ShapeVector</strong>：本质是存储容量长度为25的FVector，元素类型int64_t，存储shape信息。<a name="zh-cn_topic_0000002114052989_screen107147561639"></a><a name="zh-cn_topic_0000002114052989_screen107147561639"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000002114052989_screen107147561639">namespace op {
constexpr uint64_t MAX_DIM_NUM = 25;
using ShapeVector = FVector&lt;int64_t, MAX_DIM_NUM&gt;;
}</pre>
</li></ul>
</td>
<td class="cellrowborder" valign="top" width="24.892489248924893%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1671475613319"><a name="zh-cn_topic_0000002114052989_p1671475613319"></a><a name="zh-cn_topic_0000002114052989_p1671475613319"></a>aclnn/opdev/small_vector.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row1071455619314"><td class="cellrowborder" valign="top" width="25.93259325932593%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p197151563310"><a name="zh-cn_topic_0000002114052989_p197151563310"></a><a name="zh-cn_topic_0000002114052989_p197151563310"></a>OpExecMode</p>
</td>
<td class="cellrowborder" valign="top" width="49.17491749174918%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p571512566310"><a name="zh-cn_topic_0000002114052989_p571512566310"></a><a name="zh-cn_topic_0000002114052989_p571512566310"></a>用于表示算子运行模式的枚举类，定义参见<a href="OpExecMode.md">OpExecMode</a>。</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="24.892489248924893%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002114052989_p1371510561234"><a name="zh-cn_topic_0000002114052989_p1371510561234"></a><a name="zh-cn_topic_0000002114052989_p1371510561234"></a>aclnn/opdev/op_def.h</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002114052989_row147151563312"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002114052989_p147151756634"><a name="zh-cn_topic_0000002114052989_p147151756634"></a><a name="zh-cn_topic_0000002114052989_p147151756634"></a>OpImplMode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002114052989_p1671585617315"><a name="zh-cn_topic_0000002114052989_p1671585617315"></a><a name="zh-cn_topic_0000002114052989_p1671585617315"></a>用于表示算子精度模式的枚举类，定义参见<a href="OpImplMode.md">OpImplMode</a>。</p>
</td>
</tr>
</tbody>
</table>
