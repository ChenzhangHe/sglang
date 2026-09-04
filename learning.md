# SGLang 学习计划与状态记录

> 这是老师与学习者共同维护的学习档案。每次课从「当前状态」开始，结束时更新证据、疑问与下一步。计划可调整，掌握情况必须由学习者的回答或实践支持。

## 1. 当前状态：下次从这里继续

| 项目 | 记录 |
| --- | --- |
| 学习目标 | 理解 SGLang 的代码结构与设计取舍，能独立定位问题、验证方案、阅读和评审修复；逐步在核心推理与至少一个进阶领域达到专家水平 |
| 思考标准 | 借鉴 Jeff Dean 公开系统设计方法：明确工作负载与约束，估算成本，沿关键路径读代码，以实验校准，并解释规模、尾延迟、正确性与复杂度的取舍 |
| 仓库 | `/Users/steve/playground/sglang` |
| Fork | `git@github.com:ChenzhangHe/sglang.git` |
| 本计划代码基线 | `ff1285cc28d6b3e0ad19c45e8b14a5966bc95c78`（2026-09-03，`main`） |
| 计划建立日期 | 2026-09-03（America/Los_Angeles） |
| 已完成的准备 | fork 已创建；代码已克隆；老师已核对主要目录、入口和案例相关代码 |
| 最近一课 | 2026-09-03：推理基础问答、batch 性能手推、continuous batching 代码带读；阅读 checkout 为 `b810e3067`，模型代码仍沿用上述基线 |
| 已确认的学习表现 | 能解释自回归依赖、逐层 KV 缓存及重算恢复；独立算出单请求与 batch 的 100 / 400 token/s，能讨论吞吐与延迟取舍；代码部分为老师给入口带读，尚未独立定位 |
| 工作背景与目标 | 学习者自述有推理服务优化、continuous batching 与 profile 阅读经验；关注从请求到达到最后一个 token 返回的完成时间，尤其尾部表现；具体分位数、SLO、配置及 trace 未提供 |
| 未知项 | Python/异步、PyTorch/张量、完整 Transformer 计算与分布式能力尚未诊断；可用 GPU、环境、每周时间及实际负载长度分布待了解 |
| 当前阶段 | S0 部分诊断、S1 推理与性能概念已有答题证据；按学习者经验进入 S3 调度入门带读，停在 `ScheduleBatch.prepare_for_decode` 的职责与输入准备；S2 请求输入输出链仍待补，不将 S1/S3 整阶段标为完成 |
| 下一次课 | 已有历史 KV 后，最新 token 怎样成为下一轮 decode 的输入，并找到正确的 KV？先复述 batch 状态，再精读 `prepare_for_decode` 与调用处，见第 9 节 |
| 当前练习 | 待无提示复述 `running_batch` 与 `batch_to_run`；手推 `[A,B,C]` 中 A 结束后的请求/长度/索引对应关系，以及输入 `A B`、采样出 `C` 后下一轮的输入与 KV |
| 实验状态 | 本课为对话问答、纸上算例和静态代码阅读；未安装依赖、加载模型或运行测试/GPU 实验；学习者描述了工作中的 profile 现象，老师尚未查看实际 trace |

老师准备材料 ≠ 学习者已掌握；文档写好 ≠ 课程已完成。不要预填掌握率，也不要根据本机路径推断 GPU 条件。

## 2. 如何教：先问题，再边界，再代码，再证据

主线是一个具体场景：两个用户同时请求生成文本，其中一个 prompt 很长，另一个与历史请求共享前缀。沿着请求的生命周期逐次增加问题：怎么接入、谁先执行、缓存什么、如何算出 token、何时释放资源、如何证明正确与更快。

每次课建议 45–60 分钟，可拆成两次。一次只解决一个核心问题、精读 1–3 个函数；下面列出的文件是分多课使用的入口，不是一次读完的作业。进度按验收推进，不按周数强制翻页。

固定课堂结构：

1. **回忆 5 分钟**：学习者先解释上节内容，老师记录误区。
2. **问题 5–10 分钟**：先用具体请求说明现象、预期行为和代价，不先堆术语。
3. **图解 5–10 分钟**：说明组件责任、输入输出、状态归属；必要时画时序图。
4. **带读 15–20 分钟**：从接口与调用者进入，再读实现；遇到支线先记到停车区。
5. **预测与练习 10 分钟**：先预测结果，再手推、读测试或运行小实验；老师先给提示，再给答案。
6. **复述与记录 5 分钟**：用自己的话解释，并回答一个变化场景；更新本档案。

讲每个组件时固定回答：它解决什么问题？谁调用它？输入输出是什么？持有什么状态？必须保持什么不变量？会在哪里出错？哪一项测试或测量能证明判断？

进一步追问：若从零设计，你会先实现什么最简单的版本？当前实现增加的每一层复杂度，换来了什么可验证的收益？读函数的最终产物应是一条有适用条件的设计判断。

脚手架逐步撤除：第一次老师完整示范 → 第二次共同追踪 → 第三次只给入口与问题 → 最后由学习者独立定位。每隔 2–3 课回到完整请求链，讲清新组件与之前内容的关系。

每课记录提示等级：完整示范 / 给入口共同追踪 / 仅给问题 / 独立完成。只有能在更少提示下完成同类任务，才能提升独立定位等级。案例库是老师备课资料；案例 B/C/D 的候选方案在学习者写下预测与假设后再展示，看到现成入口不能算独立找到。

掌握等级：**未评估 → 学习中 → 能解释 → 能定位 → 能验证 → 能迁移/评审**。升级必须记录回答、图、代码定位、测试结果或 review；过关后一到两次课安排无提示复述。若卡住，补一个前置小课，再回主线。

### 2.1 Jeff Dean 的公开方法如何成为这门课的标准

以下以他的公开演讲和论文为依据；将这些方法转成 SGLang 的阅读顺序、练习与验收，是老师的应用推演，不代表 Jeff Dean 本人对这个仓库的评审或私人思考过程。

- **估算、测量、边界与规模**：在 *Designs, Lessons and Advice from Building Large Distributed Systems* 中，他强调先估计设计成本，再用 microbenchmark 建立直觉；理解基础构件、为增长设计，同时避免为假想需求过度建造。参考演讲第 23–35 页，旧硬件数字只作历史例子。[演讲原稿，LADIS 2009](https://www.cs.columbia.edu/~martha/courses/4130/au12/dean-keynote-ladis2009.pdf)
- **用户体验与尾延迟**：Dean 与 Barroso 的 *The Tail at Scale* 讨论系统规模、利用率与延迟波动如何影响交互服务，支持我们同时关注中位数与高分位数。[论文，2013](https://research.google/pubs/the-tail-at-scale/)
- **计算需求与硬件一起考虑**：Dean 的 ISSCC 配套论文讨论深度学习进展对计算设备设计的要求。我们据此把模型结构、精度、数据移动、执行后端放在一起分析。[论文，2019／ISSCC 2020](https://arxiv.org/abs/1911.05289)

这些材料由老师按课选段使用；不要求先读完论文再开始代码课。下面的问题与例子是为本课程制定的操作方法。

### 2.2 读 SGLang 的顺序：带着可检验的问题深入

第一遍建立全景，只做职责图。之后按以下循环选择本课的 1–3 个函数；无需把整个仓库从头到尾读完。

1. **写工作负载和成功标准**：短聊天还是长文档？输入/输出长度、到达速率、并发、共享前缀、模型与硬件是什么？本次要改善首 token、持续输出速度、满足延迟目标的吞吐，还是内存容量？缺少真实数据时写明教学假设。
2. **画数据、状态和等待**：沿现有请求链标出 token、KV、张量在哪里，谁持有/释放，哪里排队或同步；把启动配置与每次请求/每步 decode 的工作分开。
3. **先写一个预测**：哪项成本最可能主导？输入或并发翻倍会怎样？用纸笔算量级，注明单位与未知项；入门阶段只要求定性预测。
4. **按假设找最少的代码**：排队假设从 `scheduler.py::get_next_batch_to_run` 看起；缓存容量从 `memory_pool.py` 与 allocator 看起；执行成本从 `model_runner.py`、模型层和 attention 后端逐层下钻。先核对路径实际被配置启用，再读实现；文件均沿第 3 节目录地图定位。
5. **找能推翻预测的证据**：一个输入差异、一个状态边界、一段 trace 或一个最小 benchmark。提前写“出现什么结果时，我会放弃当前假设”；测量与预测不符时优先检查模型和测量口径。
6. **比较简单方案与当前设计**：如独立请求 vs continuous batching，重复计算 vs prefix reuse，eager vs graph。明确收益、额外状态、失败模式和适用范围，再提出方案；性能案例到 S6 必须讨论端到端收益上限。
7. **改变一个条件再判断**：请求量或上下文长度增加 10 倍、前缀不再重复、GPU 变多或有一张变慢时，哪个假设失效？这是后续迁移练习，不预设当前设计一定错误。

每课只启用当前阶段需要的步骤；其余作为老师的备课检查。S1–S2 不要求性能模型，S3 每次引入一个数量关系，S4–S5 先做带单位的容量估算，S6 起再要求时间估算与实测对照；无 GPU 时沿分析轨记录待验证项。

### 2.3 把思考标准嵌入原有课程

| 阶段 | 在原练习上增加的一个问题 | 可观察的学习证据 |
| --- | --- | --- |
| S1 | 单请求运行正确之后，多用户服务还要满足什么目标？ | 用自己的话给出一个具体目标和请求场景 |
| S2 | 请求经过边界后，哪些语义必须保持？ | 找出同义输入，并预测在哪个边界可能不再等价 |
| S3 | 增大 batch 帮助了谁，又让谁等待更久？ | 简单调度时间线，标出排队与执行；S6 再测量 |
| S4 | 缓存用多少内存，什么情况下值得保留？ | 带假设的容量估算，加一例复用有效和一例几乎无复用 |
| S5 | 一步计算中张量多大，哪些数据必须移动？ | 一层的 shape/字节表，标出物理布局尚未知的部分 |
| S6 | 局部快了，用户能快多少？ | 成本估算、局部和端到端对照，以及尾延迟/正确性证据 |
| S7 | 通信或慢 rank 会吃掉多少并行收益？ | 两卡切分与通信图；定性分析先行，实际结果另记 |
| S8 | 所选优化增加了什么状态或数据约束？ | A 用接受/回滚表，B 用路由与分发约束，C1 用精度与容量对照，C2 用驻留/传输/释放表；分别设计失败测试 |
| S9 | 若流量增长 10 倍，这个设计如何演进？ | 一页设计评审：目标、证据、方案、取舍和未解问题 |

### 2.4 三个逐步开放的数量级练习

这些都是自设教学例子，不是某个真实模型或设备的测量结果，也不是 Jeff Dean 的原话。公式只在对应前置知识通过后使用，一次选一个。

- **S4：先算 KV 容量。** 对每层具有相同 KV heads/head dimension 的普通 dense attention，未压缩、无共享、无分片时，逻辑 KV 字节数约为 `2 × 层数 × 已缓存 token 总数 × KV heads × head_dim × 每元素字节数`，2 表示 K 和 V。设 32 层、8 个 KV heads、head_dim=128、每元素 2 bytes：每 token 为 128 KiB；4096 tokens 为 512 MiB。学习者先算，再追 pool 分配。物理预留容量、padding、元数据和临时 buffer 另计；prefix 共享、TP、SWA、MLA、Mamba、量化不能机械套这个公式。
- **S6：先算搬运下界。** 假设某步必须从指定内存层移动 8 GB 数据，该层带宽上限为 1 TB/s，采用十进制单位，则单是数据搬运的理想时间下界为 8 ms。若测到 5 ms，先查实际流量、缓存复用、计时范围与单位；若测到 30 ms，查有效带宽、计算、同步等成本。知道峰值带宽不能直接推断实际耗时，也不能把可重叠的时间一律相加。
- **S6：先算端到端收益上限。** 在固定工作负载、串行分解且其他成本不变的简化例子中，一项操作占总耗时 20%，它加速 2 倍，则整体加速为 `1 / (0.8 + 0.2/2) ≈ 1.11`。先判断优化是否值得，再测排队和重叠引入的新影响；真实在线服务不一定符合这个简化模型。

从 S4 起逐步建立自己的“数字笔记”：对象、字节量、延迟/带宽、来源、硬件/配置、估算或实测、日期。未知可留空。更新硬件后重新校准，不背旧演讲中的绝对数值。

### 2.5 老师怎样追问，怎样判定你学会了

- 回答“缓存让它更快”时，先问“复用了哪一段工作？代价是什么？”，S4 再要求估算内存，S6 再要求测时间；低前缀重复率的场景下重新判断。
- 回答“加更多 GPU”时，S7 先画切分和通信，再问“如果一个 rank 慢了，哪些请求在等它？”
- 回答“修这一行就好了”时，要求说明状态不变量，以及哪个测试会在修复前失败、修复后通过；再找一个不该改变的正常行为。
- 回答“这个函数看懂了”时，请闭卷解释：设计为何存在、依赖什么假设、怎样判断对错；未学到的成本项允许标未知。

终期标准是：面对一个新问题，能**定义约束 → 提出带条件的预测 → 定位代码 → 用证据修正判断 → 解释方案取舍**。早期按阶段完成其中一部分即可；不能靠记住函数名、复述老师结论或堆高级术语升级。

## 3. 第一张地图：SGLang 在做什么

模型 `forward()` 负责一次计算。面向多用户的推理服务还要接收和转换请求、安排执行顺序、组织批次、管理有限内存、逐步生成并返回结果。后续性能优化都要同时回答「节省了什么成本」和「增加了什么状态或约束」。

先看 **Python HTTP、普通自回归文本生成、单 worker/单卡概念路径**。以下箭头表示逻辑数据流；实际存在进程通信与异步返回，不是一个同步调用栈，也不覆盖所有后端。

```text
客户端 messages / prompt
  → HTTP / OpenAI 接口：接收、校验、转换请求
  → TokenizerManager：准备 token IDs、跟踪请求与返回状态
  → Scheduler：接收请求，选择 prefill/decode 批次
  → TpModelWorker → ModelRunner → model / attention backend：执行计算
  → 采样与结果处理：产生下一批 token IDs、更新请求状态
  → DetokenizerManager：增量解码文本
  → TokenizerManager → HTTP 流式或完整响应 → 客户端

Scheduler / ModelRunner / attention backend
  ↔ prefix cache、slot allocator、KV memory pool
```

先分清：**模型权重**是模型参数；**KV cache**保留 attention 的历史 K/V；**prefix cache**索引可以复用的前缀及其缓存位置；**allocator**分配位置；**pool**保存实际张量。第一次只懂责任，S4 才深入生命周期。

| 目录 | 在整体中的作用 | 阅读时机 |
| --- | --- | --- |
| `python/sglang/cli/`、`python/sglang/launch_server.py` | 命令与启动分派；当前 CLI 推荐 `sglang serve`，旧模块入口仍保留 | S1 |
| `python/sglang/srt/entrypoints/` | HTTP、OpenAI 兼容接口、Engine 等入口 | S1–S2 |
| `python/sglang/srt/managers/` | 请求管理、调度、worker、输出；职责已部分拆到 `scheduler_components/` | S2–S3 |
| `python/sglang/srt/mem_cache/` | 前缀复用、分配、物理缓存、分层缓存 | S4 |
| `python/sglang/srt/model_executor/`、`models/`、`layers/` | 执行批次、模型结构、attention/算子与后端 | S5–S6；后两项位于 `srt/` 下 |
| `python/sglang/kernels/` | GPU 等底层算子，包含 JIT 与 AOT 相关实现 | S6，先学接口与一个算子 |
| `python/sglang/srt/distributed/`、`disaggregation/`、`speculative/` | 分布式、prefill/decode 分离、推测解码 | S7–S8 |
| `python/sglang/srt/observability/`、`python/sglang/benchmark/` | 指标、追踪、性能实验 | S1 认识指标，S6 深入 |
| `test/registered/`、`test/manual/`、`benchmark/` | 自动化测试、手动验证与专项 benchmark | 全程按当前问题选取 |
| `python/sglang/lang/` | 前端程序表达，与服务运行时区分 | S9 全景扩展 |
| `python/sglang/srt/multimodal/`、`python/sglang/multimodal_gen/` | 多模态输入处理与独立的图像/视频生成等路径 | S9，分开学习 |
| `sgl-model-gateway/`、`rust/`、`experimental/` | 网关、Rust 组件和实验性方向；逐项核对实际接入方式 | S9 |
| `docs/`、`examples/` | 文档与使用示例 | 每课按需查阅 |

目录表是查询用地图，不是首课阅读清单。首课只看主请求链上的入口、请求/调度管理、模型执行三个区域；其余随课程逐步展开。

**版本提醒**：本地 `mem_cache/README.md` 描述了分层设计与目标布局，但其中的 `pool/` 在本基线不存在；当前主要实现仍在 `memory_pool.py`、`deepseek_v4_memory_pool.py` 等文件。用真实符号定位核对文档，不能把重构蓝图当作已完成实现。当前 kernels 入口也以本地 `python/sglang/kernels/` 为准。

## 4. 循序渐进的课程路线

每个阶段包含「必修主线」；进阶专题在通过前置验收后才进入。代码路径均相对仓库根目录。

### S0 · 起点诊断与必需基础（1 课，按需补课）

- **先问什么**：你能否追踪 Python 函数/类、异常与生成器？知道张量 shape、token、模型权重吗？是否使用过异步、GPU 或分布式？
- **只补当前所需**：缺 Python 就用一个小请求处理函数练习；缺模型知识就先讲 token → embedding → attention/MLP → logits → 采样。不要求先推导 Transformer 全部公式。异步与进程在 S2 前补，CUDA 在 S6 前补，collective 在 S7 前补。
- **代码入口**：`README.md` 的 About；只浏览 `python/sglang/` 一级目录。
- **小练习/验收**：用几句话区分训练与推理、模型与推理服务。根据回答决定补课或进入 S1；未知不记成不懂。
- **低压力微任务**：老师提供 8–12 行 Python 小例子，请指出调用者、输入和返回值；再用一张 token/张量示意图解释长度或 shape。按实际解释选择补课，不只依赖“我会 Python”的自述；遇到生词先解释再试。
- **环境**：确认 Python/依赖、OS、GPU 与远程资源。先读代码和手推；本次计划不要求装环境。

### S1 · 为什么需要推理运行时（2 课）

- **问题**：单次模型计算变成多人聊天服务，多出了哪些工作？长 prompt 为什么影响首 token？
- **新概念**：prefill（处理输入上下文）、decode（沿上下文逐步生成）、TTFT（首 token 延迟）、吞吐、每 token 延迟；同时区分客户端测量与服务端时间。
- **阅读顺序**：`python/sglang/cli/serve.py` 的 `_run_llm` → `python/sglang/launch_server.py` 的 `run_server` → `python/sglang/srt/entrypoints/http_server.py` 的 `launch_server`；先知道启动分派，不展开模型加载。
- **练习**：画单请求时间线，再放入第二个请求。把等待、输入处理、prefill、decode、传输分开。
- **验收**：不看图说明完整请求的大步骤；解释 prefill 与 decode 做什么，以及吞吐提高为什么不保证每个用户等待更短。
- **暂缓**：调度算法实现、attention kernel、并行通信。

### S2 · 一条请求的输入与输出（4–6 课，先完整响应再 streaming）

- **前置**：S1；补齐 JSON/schema、生成器、`async` 和进程通信的最小概念。
- **问题**：外部 `messages` 如何变成内部生成请求？结果怎样找到原来的客户端？请求格式错误为何可能发生在模型运行之前？
- **阅读拆分**：① `srt/entrypoints/openai/protocol.py`、`serving_chat.py` 与 `srt/parser/conversation.py::generate_chat_conv`；② `srt/managers/io_struct.py`、`tokenizer_manager.py::generate_request`；③ `detokenizer_manager.py::handle_batch_token_id_out` 与返回处理。以上 `srt/` 均指 `python/sglang/srt/`。
- **桥接**：指出聊天模板与 tokenization 的区别；`conversation.py` 是其中一条模板路径，并非所有模型都必经。找到进程发送/接收边界，画返回箭头。
- **分课与负荷**：S2.1 只讲输入 schema 与模板；S2.2 讲普通完整响应，暂把 Scheduler 视为“输入 token、返回 token”的黑盒；S2.3 老师带读案例 A；S2.4 半独立做案例 B。只有能解释普通响应后，才用另 1–2 课讲生成器、IPC 与 streaming；取消/断连的资源清理在 S4 学过缓存生命周期后回访。
- **练习**：给 request ID、文本、token IDs、输出文本各找一个持有者；手推 string 与 text-parts 两种输入。流式课再加入结束标志、取消与异常。
- **案例**：先跟老师读已合并 PR #35915，再半独立分析 issue #37845（见案例库）。
- **验收**：解释一次“HTTP 输入合法但模板处理失败”的路径，并给出最小测试和一个应保持不变的行为。
- **暂缓**：完整阅读巨大的 manager 文件；多模态、tool calling 的全部规则。

### S3 · 多个请求怎样组成执行批次（3 课）

- **前置**：S2；S1 的 prefill/decode；先用简单预算理解每个请求会占缓存空间。
- **问题**：为什么不能每个请求独占 GPU？新请求、生成中的请求和长输入如何共享执行机会？
- **阅读拆分**：① `python/sglang/srt/managers/scheduler.py::event_loop_normal`；② `get_next_batch_to_run` 与 `schedule_policy.py`；③ `schedule_batch.py` 的 `Req`、`ScheduleBatch`、`NextBatchPlan`，再看 `run_batch`、`process_batch_result` 的接口。
- **练习**：纸上模拟 A 长输入、B 短输入、C 中途到达，逐步列等待队列、正在执行的请求、token 预算和已结束请求。明确这只是教学策略，之后与真实代码决策比较。
- **验证方式**：本阶段以纸上预算表与真实 `get_next_batch_to_run` 的入口追踪为主，不要求构造完整 Scheduler 测试夹具。`test/registered/unit/managers/test_scheduler_chunked_req_gate.py` 涉及 SWA（滑动窗口 attention）与缓存状态，移到 S4 后作为选读，届时先补窗口与普通完整上下文的区别。
- **验收**：解释 continuous batching、chunked prefill 的动机；能用一个场景说明吞吐、延迟、公平性的取舍，并定位批次生成与结果更新的责任。
- **暂缓**：`event_loop_overlap`、PD、speculative 分支；普通循环是教学起点，不声称它是所有配置的默认执行方式。

### S4 · KV cache 的正确性与生命周期（3–4 课）

- **前置**：S3；补 attention 中 Q/K/V、序列长度、缓存占用的简单量纲。
- **问题**：为什么 decode 保留历史 K/V？相同前缀如何复用？请求结束或缓存不足时，哪些位置可以释放？
- **阅读拆分**：① `python/sglang/srt/mem_cache/radix_cache.py` 的 `match_prefix`、`insert`；② `evict` 与 `base_prefix_cache.py` 的接口；③ `allocation.py`、`allocator/`、`memory_pool.py` 的责任边界。通过单一实现的验收后，才另开选读比较 `unified_radix_cache.py` 与 `unified_cache/`，不计入首轮必读。
- **练习**：用 `[A,B,C]` 与 `[A,B,D]` 画共享前缀；给缓存节点、引用/锁定状态、slot 归属和可驱逐状态做一张表。解释取消、完成与重复请求的资源变化。
- **测试入口**：`test/registered/unit/mem_cache/test_radix_cache_unit.py`，只选一个前缀匹配或驱逐案例。
- **回访**：带着资源归属图返回 S2，解释请求取消/断连时谁终止计算、谁释放槽位；正常结束、取消与可复用前缀不可混为一谈。
- **验收**：区分逻辑索引与物理内存；能说明为什么仍被使用的缓存不能驱逐，为什么不同 dtype/layout 会改变容量。不能只背“RadixAttention 很快”。
- **案例**：#37852 先做指标追踪练习；复杂 DeepSeek 布局的完整修复推迟到 S8 后。
- **暂缓**：第一次不读 HiCache 全套后端，不把 Mamba 状态等同于普通 KV。

### S5 · 从批次到模型的一次 forward（3 课）

- **前置**：S4；补 PyTorch shape、矩阵乘、attention、logits、采样。
- **问题**：调度器的请求集合怎样变成设备上的张量？同一个模型为什么有不同执行后端？
- **阅读拆分**：① `python/sglang/srt/managers/tp_worker.py::TpModelWorker.forward_batch_generation` → `model_executor/forward_batch_info.py::ForwardBatch`；② `model_executor/model_runner.py::forward` 及其实际委托；③ `models/llama.py` 中 `LlamaForCausalLM` → `LlamaModel` → 一个 `LlamaDecoderLayer`，再定位 `layers/radix_attention.py`、`layers/sampler.py`。本段后续路径均在 `python/sglang/srt/` 下。
- **练习**：记录 prefill/decode 的 token、position、sequence length、cache location 各代表什么；手画一层的张量变换，区分模型逻辑和服务元数据。
- **验收**：能从 scheduler 追踪到一个模型层，说明 cache location 为什么影响 attention 正确性；知道 logits 与最终 token 的区别。
- **暂缓**：先选一个 dense 模型；不要同时展开所有模型、量化与 MoE。

### S6 · 性能分析：先测量，再解释优化（3–4 课）

- **前置**：S5；补 CPU/GPU 异步、kernel launch、内存带宽与计算吞吐、同步点；开始统计实验前补百分位数、样本量、到达速率与并发的区别。
- **问题**：慢发生在排队、CPU 调度、算子执行还是传输？overlap/CUDA graph 优化的是哪段成本？
- **阅读拆分**：① `python/sglang/benchmark/serving.py`、`srt/observability/req_time_stats.py`；② `srt/managers/scheduler.py::event_loop_overlap`；③ `srt/model_executor/runner/eager_runner.py` 对照 `decode_cuda_graph_runner.py`；④ 一个 `srt/layers/attention/` 后端与它调用的 kernel 接口。这里 `srt/` 均在 `python/sglang/` 下。
- **实验**：先写工作负载分布、到达方式与成功标准，再固定 commit、模型、硬件、dtype、输入/输出长度与并发；先 warmup，再重复采样。每轮只改一个变量，同时记录局部耗时、端到端 TTFT、每 token 延迟、吞吐和波动；按请求数与样本量报告 p50/p95/p99，样本不足不强行解释 p99。说明 offered load（发起的负载）、完成量、错误/超时与是否达到延迟目标，避免只对完成的快请求报喜。没有 GPU 时由老师提供有来源的 trace 或标注为模拟的时间线。
- **双轨验收**：无 GPU 时交一页分析报告：瓶颈假设 → 带来源的已有 trace/测量或明确标为模拟的时间线 → 判别实验设计 → 尚待验证结论，可标“能解释/能定位”。有兼容 GPU 时再交一页实测报告：假设 → 复现命令与原始结果 → 对照 → 正确性检查 → 波动与适用范围，才可升级“能验证”。资源不足不阻断后续概念学习，但保留实测能力缺口。不能把一次吞吐数字或更低 TTFT 当作普遍结论。
- **延伸**：看一个算子的参考实现、测试、优化实现，理解访存/并行化后再深入 CUDA/Triton。

### S7 · 多 GPU 与分布式服务（3–4 课）

- **前置**：S6；先补 rank、collective、all-reduce/all-gather、通信开销与失败传播。
- **问题**：单卡放不下或吞吐不够时怎么分工？为什么更多 GPU 不一定更快？
- **阅读拆分**：① `python/sglang/srt/distributed/parallel_state.py`、`communication_op.py` 与 `layers/linear.py` 的张量并行；② `managers/data_parallel_controller.py` 与 `scheduler_pp_mixin.py`；③ `disaggregation/` 的 prefill/decode 分离。除完整路径外，均相对 `python/sglang/srt/`。
- **练习**：先手推两卡矩阵切分与一次通信，再比较 TP（切张量）、DP（请求/副本层面分工）、PP（切层）；最后画 PD 的请求与状态转移。
- **拆课边界**：先完成 TP，再认识 DP/PP；PD 是分离推理阶段，不与前三者视为同一种切分。通过 TP/DP/PP 的概念验收后，PD 单独安排后续专题课与状态传输练习。
- **验收**：说清每种方式分割什么、增加什么通信和状态；能为容量或延迟问题选择一个方案并指出限制。
- **硬件门槛**：图解与代码阅读可本地做；实际多卡验证必须记录拓扑、模型、配置与运行结果，未运行则保留待验证。

### S8 · 核心高级专题与综合正确性（首个专题约 4–6 课，其余另行排课）

- **前置**：S4–S6；涉及分布式时加 S7。先完整学一个专题，不在 4–6 课内并行完成 A/B/C。这个估计不含缺失基础的补课、环境搭建与真实后端实验时间。先做有来源的静态推理，GPU 验证单独记录。
- **顺序 A**：普通 KV → hybrid attention/Mamba 状态 → draft/verify/accept/rollback → checkpoint 一致性。先完成普通 speculative 的接受/拒绝小例子，再进入 #37817。代码从 `python/sglang/srt/speculative/spec_utils.py`、`python/sglang/srt/speculative/dflash_worker_v2.py` 与 `python/sglang/srt/mem_cache/mamba_radix_cache.py` 进入。
- **案例**：#37817。先手推 255→256 边界，再追 accepted length、状态复制与 scheduler 更新；不能看到一个可疑表达式就宣称修复完成。
- **顺序 B**：dense 模型 → MoE 路由 → expert parallelism → 负载均衡；入口 `python/sglang/srt/layers/moe/` 与 `eplb/`（后者在 `srt/` 下）。
- **顺序 C**：权重/KV dtype → 量化误差与内存 → HiCache 分层传输；入口 `python/sglang/srt/layers/quantization/`、`mem_cache/hiradix_cache.py`、`mem_cache/storage/`（后两项在 `srt/` 下）。回访 #37852 的复杂布局。
- **验收**：独立完成一个专题的失败场景、状态不变量、候选方案与回归矩阵；区分数学小例子成立、单元测试通过与真实后端端到端通过。
- **深广顺序**：初次建议选 A，因为它能回用 S4 的状态与缓存知识；先补 hybrid/线性状态小课，再引入 speculative。若学习者更关注部署，可选 C，但量化与 HiCache 分成两个独立单元。剩余专题留在 S9 长期循环，逐项补齐概念地图，不能因选修一个方向就宣称全部高级能力已掌握。

### S9 · 全景扩展与专家实践（长期循环）

- **扩展地图**：依次了解 structured output / tool parser、LoRA、multimodal、diffusion、网关、Rust、其他硬件和 rollout/权重更新。相关入口是 `python/sglang/srt/constrained/`、`function_call/`、`lora/`、`multimodal/`、`weight_sync/`，以及地图中的 `multimodal_gen/`、`lang/`、`sgl-model-gateway/`、`rust/`；缩写路径相对 `srt/`。
- **深度要求**：每个方向能解释职责与主线关系；选择至少一个方向完成真正的实现/测试/评审深潜。专家能力靠持续实践积累，不以“读遍所有文件”为验收。
- **综合任务**：独立选一个当前 issue，定位、复现、提出两种方案、实现本地最小修改、验证边界与性能风险，并形成可评审的变更说明。另独立 review 一个别人的修复，指出测试能证明和不能证明的内容。附一页设计判断：工作负载与目标、最简单基线、主要成本、证据、为何选择当前方案，以及一个规模增长 10 倍或局部失败后的变化场景。
- **完成标准**：能解释端到端架构；跨至少两个组件定位问题；给出可复现证据；说明设计取舍；遇到新 issue 能制定验证路线。GPU/分布式实践未完成时保留对应能力缺口。

## 5. 第一轮真实案例：先读解答，再自己提出方案

以下为 **2026-09-03（America/Los_Angeles）通过 GitHub 查询的快照**。三个 issue 当时为 OPEN；进入案例课时重查正文、评论、关联 PR 与最新代码。OPEN 不等于尚无人处理；关闭也不等于在所有版本中已修复。这里是教学选题，不是本轮要修复的任务。

### A · 已合并示范：空 assistant 消息（S2）

- **来源**：[PR #35915](https://github.com/sgl-project/sglang/pull/35915)，2026-08-24 UTC 合并，merge commit `0c1e9bda57732fb7276aeec36fac5f22ec97aa01`。
- **现象/方案**：PR 报告 Mistral tokenizer 拒绝空 assistant turn；补丁在交给对应 tokenizer 前移除没有内容且没有 tool calls 的 assistant turn，保留真实内容和工具调用。
- **本地入口**：`python/sglang/srt/utils/hf_transformers/mistral_utils.py::patch_mistral_common_tokenizer`；`test/registered/unit/tokenizer/test_mistral_empty_assistant.py`。
- **教学方式**：先只看失败输入，预测哪里应当校验；再读测试和补丁，解释为什么不能删除所有空字符串或所有 assistant 消息。
- **迁移限制**：这个 Mistral tokenizer 适配路径与案例 B 的 `generate_chat_conv` 路径不同；迁移的是定位与边界测试的方法，不能直接把过滤补丁搬过去。
- **证据边界**：已核对合并状态与本地实现/测试；PR 的服务验证是作者报告，本学习环境尚未运行。

### B · 初级开放案例：相同文本拆成多个 part 后失败（S2）

- **来源**：[issue #37845](https://github.com/sgl-project/sglang/issues/37845)，查询时 OPEN；已有评论者表示愿意修复。
- **问题**：报告中 string 输入成功，但 `system`/`assistant` 的多个 text parts 被拒绝；这不需要先研究 GPU。
- **入口**：`python/sglang/srt/entrypoints/openai/protocol.py` → `python/sglang/srt/parser/conversation.py::generate_chat_conv` → `test/registered/unit/parser/test_conversation.py`。
- **已核对**：本地该模板路径对 system/assistant list content 的长度有等于 1 的限制。尚未运行复现；不推广为所有模板/后端都失败。
- **候选方案，待验证**：按顺序拼接允许的 text parts，同时维持非文本部分的校验；这是 issue 预期与讨论方向，不是已确认合并的修复。
- **学习者任务**：先写 string、单 part、多 part 的预期；再加入空列表、空内容、非文本和 user 角色，区分规范问题与实现问题。先写能暴露行为差异的测试，再讨论修改位置。
- **资源**：报告提供无 server/GPU/权重的复现；实际 Python 依赖是否可用需先检查，不能假定 Mac 可直接跑全套。

### C · 中级开放案例：已分配缓存，指标却是零（S4/S6，完整修复 S8 后）

- **来源**：[issue #37852](https://github.com/sgl-project/sglang/issues/37852)，查询时 OPEN。
- **问题**：报告 DeepSeek-V4 的 KV 内存指标为 0。先分辨“统计缺失”和“真的没有分配”，不要直接判断缓存算法失效。
- **入口**：`python/sglang/srt/mem_cache/memory_pool.py::KVCache`、`deepseek_v4_memory_pool.py`（同目录）。指标链是 `python/sglang/srt/managers/scheduler.py::emit_metrics_constants` → `python/sglang/srt/observability/metrics_collector.py`；内部状态另查 `python/sglang/srt/managers/scheduler_components/load_inquirer.py`。先追一个出口，再比较另一个。
- **证据边界**：本地已看到 `KVCache.mem_usage` 的零初值与常规统计入口；具体遗漏与各种布局的影响来自报告，尚未完成全路径验证。
- **候选方案，待验证**：按实际分配的物理 tensor 统计，并在组合 pool 聚合；考虑单位、共享 storage 去重、padding、不同 dtype、rank/layer 范围。先明确统计口径，再选统计实现位置。
- **学习者任务**：画“物理 buffer → pool → allocator/调用者 → 指标”链；用小型假数据设计计数测试。S4 只完成这一步，完整 backend 验证后置。
- **资源**：真实报告环境为多张 H100；无对应 GPU 时仅做静态追踪与简化计数练习，不能记成端到端已复现。

### D · 高级开放案例：推测解码跨越状态检查点（S8）

- **来源**：[issue #37817](https://github.com/sgl-project/sglang/issues/37817)，查询时 OPEN，报告者注明完整运行时验证仍在进行。
- **问题**：DFLASH + hybrid 模型的接受 token 跨越 checkpoint 边界时，线性状态与前缀缓存记录可能不一致。
- **入口**：`python/sglang/srt/speculative/dflash_worker_v2.py::_update_target_mamba_state_after_verify`，再追 `spec_utils.py`（同目录）、scheduler 结果处理与 Mamba cache。
- **已核对**：本地有报告指出的前后长度比较表达式；尚未证明所有相关路径下的根因与修复。
- **候选方案，待验证**：以 verify 前长度加实际 commit 长度推导提交后位置，检查边界与应保存的状态索引，并核对真正的状态写入及回滚。不能只改 mask 而忽略后续 tracking point。
- **学习者任务**：手推 `(pre, commit) = (255,1)、(254,2)、(250,3)、(256,1)`；随后设计 batch 中不同请求、拒绝 token、缓存再命中等验证。
- **资源**：整数/CPU 小例子可展示边界问题；模型正确性、相关 kernel 和完整 speculative 路径仍需兼容 GPU 环境验证。

## 6. 阅读当前 issues 与解决方案的方法

每阶段挑一个范围可控的案例；不要把最新/最热门的大模型部署问题直接当入门作业。优先选复现清楚、能找到相关代码/测试、前置知识匹配的案例。

选题前先说明影响：是错误输出、API 行为不一致、容量统计误导，还是尾延迟/吞吐问题？哪些负载与用户受影响，证据强度和复现成本怎样？修复优先级与教学难度分别判断；容易入门不等于生产影响最大。

1. 写下**用户可见行为**：预期、实际、最小输入，是否确为 bug。
2. 记录**版本与环境**：本地 commit、报告 commit、模型、后端、dtype、硬件、配置；比较代码差异。
3. 区分**已知事实与猜测**：报告者观察、老师静态判断、自己复现、上游测试分别记录。
4. 把问题放到架构图上：哪个边界最早出现错误？先列两个可区分的假设；性能问题另写工作量、数据量、等待与同步的粗略成本，正确性问题先写状态不变量。
5. 找最小判别实验，先预测，再观察；明确什么结果将推翻各假设。学习阶段可用手推和测试阅读作为中间步骤。
6. 比较方案：改哪层、为何在这里修、兼容性与性能代价、能否更小；指出增加的状态与维护成本。性能修复先估端到端收益和受益负载，再安排 benchmark。
7. 验证失败用例与邻近正常用例；需要时做真实后端正确性和性能对照。
8. 再读关联 PR 的 diff、测试、review、CI 与 merge 状态。识别“有人提方案”“已合并”“本地已验证”的区别。

回到现有案例时，A/B 追问“语义应在哪个边界统一”；C 追问“谁拥有物理 storage，指标是否足以支持容量决策”；D 追问“已提交长度、缓存与状态复制必须满足什么一致关系”。无需为了套用性能框架给每个解析 bug 都做速度测试。

课前可用的只读定位命令：

```bash
git rev-parse HEAD
git status --short
rg -n 'def generate_chat_conv' python/sglang/srt/parser/conversation.py
gh issue view 37845 --repo sgl-project/sglang --comments
gh pr view 35915 --repo sgl-project/sglang
```

每次更新代码后记录新基线、受影响课程与符号位置。未解决的案例若已有修复，改成“先预测修复，再对照”的复盘题。学习文件更新不会自动同步 fork 或上游，也不代表自动监控 issues。

## 7. 环境与实践分层

| 层级 | 可以做什么 | 如何记录 |
| --- | --- | --- |
| 无运行环境 | 浏览、画图、手推状态、读测试、比较 diff | 静态分析/手推，不记成测试通过 |
| 可用 Python/CPU 依赖 | 解析、纯逻辑、mock 单元测试、小型张量练习 | 记录命令、依赖、输入和实际结果 |
| 兼容单 GPU 环境 | 小模型请求、缓存、采样与性能实验 | 记录模型 revision、硬件、后端、dtype、配置与测量 |
| 多 GPU/专项后端 | TP/PP/PD、复杂模型、通信与容量验证 | 额外记录拓扑、rank 和各端版本；未具备条件的实验保持待验证 |

`test/registered/unit/` 的定义是“不启动 server、不加载权重”，并不保证每个测试都不需要 GPU。参考其 README 和具体文件的 CI 注册/import。只运行当前问题所需的测试，不以完整测试套件作为入门门槛。开始 GPU 课时再制定与实际资源相符的模型/环境方案。

## 8. 能力状态与证据

| 能力 | 当前等级 | 证据 / 最近无提示证据日期 | 下一步 |
| --- | --- | --- | --- |
| Python、张量、完整模型计算 | 未评估 | 本课未做 Python/shape/attention 计算诊断 / — | 后续按所需前置补齐 |
| 自回归生成与 prefill/decode 概念 | 能解释 | 2026-09-03：仅给问题即解释未来 token 依赖尚未确定的前序 token；逐 token 依赖成立，完整上下文与 prefill 并行由老师补充 | 下次无提示复述输入 `A B`、采样出 `C` 后的计算 |
| 代码地图、请求生命周期 | 学习中 | 2026-09-03：已听讲模型/服务职责并带读 scheduler 局部循环；完整请求图与启动链未独立复述 / — | 补 S1 全链图及 S2 输入输出链 |
| 请求解析、返回与异步边界 | 未评估 | 尚未展开 HTTP、模板、tokenizer/IPC、detokenizer 链 / — | S2 待补 |
| 调度与批次 | 学习中 | 2026-09-03：能提出请求调度、在 forward 轮次间安排工作的思路；老师带读接纳/合并/过滤，`running_batch` 与 `batch_to_run` 经追问后讲解；尚无独立定位或闭卷复述 | S3：复述三种状态，手推成员与元数据变化，再读 `prepare_for_decode` |
| KV 缓存作用与重算取舍 | 能解释 | 2026-09-03：仅给问题即指出各层 KV 需要保留，缓存丢弃后可重新 prefill 恢复；重算包含原始输入与已生成 token 的细节由老师补充 | 无提示复述恢复上下文；S4 再做容量与资源生命周期练习 |
| Prefix cache 与物理内存管理 | 未评估 | 只介绍接纳预算、撤回重排队和结束后的资源处理，未读 prefix 共享、allocator 实现 / — | S4；不由 KV 概念答对推断实现能力 |
| 模型执行与 attention 接口 | 学习中 | 2026-09-03：刚听讲常驻共享权重、最新 token、位置/长度、KV 索引和新增 KV 空间；已定位 `prepare_for_decode`，尚未逐段精读或追入模型 forward / — | 下次先补批次准备接口；模型各层留到 S5 |
| 性能测量、overlap、kernel | 未评估（实测） | 学习者自述工作中见到 decode/prefill 交替；本课未提供 trace、未进行 benchmark / — | S6；将 profile 假设与真实测量分开 |
| 分布式与 PD | 未评估 | 无实验 / — | S7；进入专题后拆分记录 |
| 推测解码与 hybrid 状态 | 未评估 | 无专题实践 / — | S8 A |
| MoE / expert parallelism | 未评估 | 无专题实践 / — | S8 B |
| 量化 | 未评估 | 无专题实践 / — | S8 C1 |
| 分层缓存 | 未评估 | 无专题实践 / — | S8 C2 |
| Issue 定位、方案与 code review | 未评估 | 只有老师选题 / — | 从 S2 开始逐步撤除提示 |
| 工作负载、约束与系统取舍 | 能解释（入门） | 2026-09-03：主动提出连续多轮 decode、等待更多请求合并 prefill；能解释 batch 使部分用户变快、部分变慢，并明确工作中关注最后 token 的完成时间尾部 | 用具体到达时间与长度分布检验策略；参数调优收益仍待测 |
| 数量级估算与测量校准 | 能解释（基础算例）；实测未评估 | 2026-09-03：独立算出 10 ms/单 token 为 100 token/s，20 ms/8 token 为 400 token/s；补充排队条件后预测首个完成时间 1 s / 2 s；最后请求 8 s 和平均 4.5 s / 2 s 由老师给出 | 下次自行复算完整时间线；S4/S6 再扩展容量、带宽与测量校准 |
| 反证、规模变化与设计评审 | 未评估 | 无迁移证据 / — | S2 预测起步，S6–S9 深入 |

进入具体专题后，继续按需要拆出独立能力行；没有证据的其他子能力保持未评估，不随一个专题整体升级。

### 2026-09-03 · 第 1 次对话课：从推理基础到 continuous batching

- **基线与证据类型**：checkout `b810e3067`（前一文档提交），代码基线 `ff1285cc28d6b3e0ad19c45e8b14a5966bc95c78`。课堂为问答、教学算例与静态带读；未运行模型、测试或性能实验。
- **学习者回答与提示等级**：基础题仅给问题，学习者解释自回归依赖、每层 KV 缓存和重做 prefill；独立算出系统吞吐 100 / 400 token/s。串行与 batch 的完成时间题在老师加入排队条件后继续推理；完整最后/平均完成时间由老师补齐。代码采用老师给入口与片段、共同追踪，不记为独立定位。
- **实际带读路径**：`scheduler.py` 的 `event_loop_normal`、`get_next_batch_to_run`、prefill 接纳与 `update_running_batch`；`schedule_batch.py` 的 `filter_batch` / `merge_batch`，以及 `prepare_for_decode` 中分配新增 KV、更新长度的片段；`scheduler_components/batch_result_processor.py` 的 token 追加、结束判定与资源处理接口。以上 manager 路径均在 `python/sglang/srt/managers/` 下；`run_batch` 只认识执行接口，模型 forward 尚未展开。
- **澄清与待复述**：长输入与长输出影响不同阶段；batch 的吞吐提升不等于单请求逐 token 更快，排队改变最终完成时间；`running_batch` 保存持续解码集合，`batch_to_run` 是本轮任务；过滤请求时必须同步长度与索引；刚采样出的 token 要在下一轮被处理后才产生自己的 KV。后面三项主要为老师讲解，需下次无提示复述。
- **性能支线与边界**：查到 `prefill_decode_interval`、`PrefillDelayer` 的可选等待逻辑，并介绍 chunked prefill 动机；未逐段精读全部实现。学习者关注请求到达到最后 token 的完成时间尾部，报告 profile 有 decode/prefill 交替；“prefill 阻塞 decode 推高尾部”只是假设，未看 trace，未确认瓶颈或调优收益。教学选择普通生成、prefill/decode 分开执行；不将 `event_loop_normal` 当成全部配置的默认路径。
- **停点与路线调整**：按学习者已有经验进入 S3 的引导阅读，S2 仍未完成；停在 `ScheduleBatch.prepare_for_decode` 的职责和输入概念，不提前记为模型执行阶段完成。下次先复述状态与 token/KV 关系，再追下一轮输入，详见第 9 节。

### 每次课的追加记录模板

日常只更新第 1 节当前状态、发生变化的能力行，再追加 3–6 行短记录（日期/commit、问题、自己的回答与提示等级、证据、误区、下一步）。下方完整模板用于复杂课或实验，不强制每课填满。第 1 节是唯一进度入口，第 9 节是随之更新的教案，开课前检查二者一致。

做设计/性能练习时，在“证据”那一行保留最短的判断链：`假设与单位 → 预测 → 实际证据 → 是否改判及原因`。静态分析就写“实际待测”，不为填表虚构数字。

```text
日期 / 课次 / 使用 commit：
本课核心问题与前置知识：
阅读文件与符号（只写实际读到的）：
学习者自己的解释 / 预测：
工作负载与目标 / 本阶段适用的成本估算 / 可推翻预测的结果：
提示等级 / 最近无提示复述结果：
老师发现的误区与纠正：
练习或实验：输入、预期、实际、命令/结果位置、运行环境：
证据类型：静态阅读 / 手推 / 单元测试 / GPU 实验 / 上游报告：
掌握等级变化及依据：
尚未解决的问题、资源限制：
需要复习的旧知识：
下一次唯一核心问题 / 1–3 个阅读入口：
```

### 问题停车区

| 问题 | 何时回来 | 状态 |
| --- | --- | --- |
| Python/张量能力、可用时间与工作负载是什么？ | 下次按所需前置简短补问 | 推理概念与批次性能有答题证据；服务优化经验为自述；具体环境、时间、输入/输出长度分布待了解 |
| S2 的完整输入输出链何时补齐？ | 当前调度小课收尾后，进入模型实现前安排桥接 | 尚未带读，不因 S3 预览而跳过验收 |
| profile 中 prefill/decode 交替是否抬高端到端完成时间尾部？ | S6，拿到工作负载、配置与实际 trace 后 | 假设待验证；用户本课选择先回调度主线 |
| 有哪些本地/远程 GPU 与 Python 环境？ | S0 确认，开始实验前细化 | 待了解 |
| cache 与 scheduler 重构后路径如何变化？ | 更新基线时 | 持续核对 |
| 哪个进阶方向最值得长期深耕？ | S5 后根据兴趣与表现选择 | 待选择 |

## 9. 下一次课的具体教案

**题目：已有历史 KV 后，下一轮 decode 还需要准备什么？**

- **起点与边界**：接上 `ScheduleBatch.prepare_for_decode`，属于 S3 的批次准备及通往 S5 的接口预览，尚未进入模型各层 forward。普通自回归、无推测解码、prefill/decode 分开执行；需要时说明真实配置分支，不把简化路径冒充全部行为。S2 与 S4 的未读部分仍需补齐。
- **先无提示复述**：A、B 正在 decode，C 被选去 prefill 时，`running_batch` 与 `batch_to_run` 分别是什么？若 A 结束，`[A,B,C]` 对应的请求、长度与槽位索引怎样一起过滤？答不清就先回看上课片段。
- **本课唯一核心问题**：输入 token `A B` 经 prefill 后采样出 `C`，下一轮输入是什么、历史 KV 覆盖哪些位置、本轮又新增什么？强调字母只是 token 的代号，权重通常已加载并共享，最新 token 的 KV 尚待计算。
- **阅读入口（最多两个函数）**：`python/sglang/srt/managers/schedule_batch.py::ScheduleBatch.prepare_for_decode` 与 `python/sglang/srt/managers/scheduler.py::Scheduler.run_batch`。先区分 `input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`，再核对最新 token 在当前版本真实的数据传递路径；不能仅凭旧注释认定它总在某个函数直接赋值。
- **纸上练习与验收**：给出两个不同上下文长度的请求，画一轮输入、KV 读取索引、新增 KV 写入位置和结果归属。能解释为什么 batch 位置变化不能打乱请求/KV 对应关系，并自己指出准备与执行的边界。`alloc_for_decode` 先视作分配接口，不同时深挖 allocator 或 attention kernel。
- **后续桥接**：本课复述通过后，补请求输入输出链和必要的 KV 索引/生命周期知识，再沿 worker / ModelRunner 进入真正的模型 forward。prefill/decode 参数调优、overlap、PD、推测解码与真实 tail latency 实验留在对应阶段，不在这节铺开。

## 10. 维护记录

- 2026-09-03：老师完成初版选题与代码核对。学习状态初始化为待评估；开放问题的方案均为候选思路，未运行修复验证。
- 2026-09-03：独立 subagent `review_learning_plan` 已完成教学审阅。结论：主线依赖合理，但初稿的 S1→S2 衔接、S2 负荷、S6 无 GPU 验收、S8 范围需要调整。
- 已采纳：保留 S1 第 2 课；S2 拆成 4–6 课，取消/资源回收后置；S6 分析与实测双轨验收；S8 首次只选一个专题。另补诊断微任务、提示等级、无提示证据日期与简短课后记录；首课只定位入口，复杂测试和分布式专题延后。修订后的主线不要求提前具备 GPU 或一次读完大文件。
- 修订版复核：同一独立 subagent 确认上述四项优先风险均已解决，未发现阻碍交付的课程顺序问题。
- 2026-09-03：根据用户的 Jeff Dean 标准补充公开资料来源与教学应用：工作负载/约束、数量级估算、数据与状态流、反证实验、尾延迟、增长与复杂度取舍。将这些要求纳入阶段练习、issue 分析、终期评审与证据记录；原有学习状态保持未评估，数字例子均为教学假设。以上既有审阅结论对应此前版本，本次增补另行复核。
- 本次增补复核：独立 subagent 确认来源归属、递进安排和数值例子合理；已按建议区分 S8 各专题证据，补充 S6 的统计前置，并统一容量/时间估算的学习时机。

- 2026-09-03：根据第 1 次对话课更新当前状态、逐项能力证据、课堂记录、停车区和下一课教案；区分独立回答与老师示范，保留 S2/S4 等未读项，停点设为 `prepare_for_decode`。本次仅维护学习文档，未改动运行时代码、未进行性能验证。
