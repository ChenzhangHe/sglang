# SGLang drills：从 batch 到 attention 的 KV 读写

日期：2026-09-19。代码核对基线：`a1ee1ca90531bebdedba70f4ed5ff15179d162c1`。

目标：能解释模型与运行时怎样通过 metadata 协作，并能独立在代码里找到依据。限定普通单卡、dense attention、每请求一个 token 的 decode；暂不展开 speculative、PD、量化、CUDA graph、滑动窗口。

每次只做一道，建议 10–20 分钟。顺序是 **先预测 → 写理由 → 读代码 → 修正判断**。不需要安装 SGLang 或 GPU。先把答案写进 `answers.md`，再看 `solutions.md`。下列代码均是教学伪代码，真实源码通过各题给出的符号定位。

状态：练习已准备，尚未验收。对话中已答对五个 batch 字段，能识别单独交换 token 导致请求错配，并理解 batch 重排无需搬动历史 KV；这不等于下面的源码定位练习已完成。

## 下次从这里开始：编写迷你 decode attention

2026-09-19 停点：已理解 batch metadata 与模型 attention 的协作，下一次由学习者实际实现四个函数。练习代码仍未完成，不记作已掌握或已通过测试。

- `mini_attention.py`：填写 `KVPool.write`、`KVPool.read`、`build_history_indices`、`decode_attention` 四处 TODO；`attend` 数学函数已提供。
- `test_mini_attention.py`：10 个标准库 unittest，覆盖 slot 顺序、跨层隔离、ragged 索引、当前 KV 写入、请求隔离、重排和非均匀 attention。
- `answers.md`：记录自己的预测、实现过程和测试证据。
- `solutions.md`：纸上练习参考答案；实现前先不要展开 Drill 3。

从仓库根目录运行（Python 3.9+，无需 pip、PyTorch 或 GPU）：

```bash
python3 -B -m unittest discover -s ai_understanding -v
```

初始状态应为 **1 个数学 helper 测试通过，9 个练习测试因 NotImplementedError 报错**。这表示待填写的题目被测试覆盖，不表示已经实现。建议每次只完成一个阶段：

1. 实现 write/read，运行 `python3 -B -m unittest discover -s ai_understanding -k TestKVPool -v`。
2. 实现历史索引，运行同一命令并把 `TestKVPool` 换成 `TestHistoryIndices`。
3. 先手算 README 的 Drill 4，再实现 decode，运行全部测试。
4. 所有测试通过后，解释为什么换 batch 顺序无需移动历史 KV，再做 Drill 6 源码追踪。

这里只模拟单 head、head_dim=1，输入是已经计算好的 Q/K/V；layer_id 放在教学 batch 中便于调用，真实 SGLang 由 attention layer 提供。扩展到向量、多 head 与真实 kernel 放到下一阶段。

## 统一场景

| 执行前 | A | B |
| --- | --- | --- |
| 历史 KV token 数 | 3 | 2 |
| 本轮输入 token ID | 42 | 55 |
| request row | 7 | 9 |
| 历史 slots | [11,18,23] | [60,72] |
| 本轮分配的新 slot | 101 | 205 |

当前 token 已在上一轮被采样，但它自己的 KV 尚未计算。完成本轮 allocation 后，request row 的有效映射包含新 slot；写入发生在对应层计算出 K/V 后。

## Drill 1 · 字段的含义与时间边界

在 `prepare_for_decode` 完成、模型计算开始前，写出 `input_ids`、`positions`、`out_cache_loc`、`req_pool_indices`、`seq_lens` 的值与 shape。

回答：

1. `seq_lens` 包含当前 token，是否表示当前 token 的 KV 已经写好了？
2. token ID、position、request row、KV slot 四种编号有什么区别？
3. 本轮要生成几组“当前 token 的 K/V”？如果模型有 32 个 attention 层呢？先以每个 token、每层一组 K/V 计数，不展开 heads。

源码核对：`ScheduleBatch.prepare_for_decode` 与 `ForwardBatch.init_new`。找出分配位置、长度递增和 position 推导的表达式，各记一个符号/行号。

过关：值与时间边界正确，能区分预留槽位与写入数据。

## Drill 2 · Shape 都正确，为什么结果错了？

有人为了把 batch 排成 `[B,A]`，只做了：

```python
input_ids = [55, 42]
# 其他字段保持 Drill 1 的值
```

1. 写出下标 0 的 token 会使用哪个请求的 position、写入哪个 slot、读取谁的历史。
2. 列出正确的五个向量。
3. `req_to_token` 的 row 7 与 row 9 需要交换吗？KV pool 中的数据需要搬动吗？
4. 设计一个能抓到这种错误的测试。只断言所有向量长度等于 2 够不够？

加题：B 完成并退出后，剩下 A，写出五个向量。这里只练过滤 metadata，暂不模拟缓存释放。

过关：能陈述“同一下标对应同一请求”的不变量，测试可区分正确与错误实现。

## Drill 3 · 写一个最小 cache-aware attention 接口

先比较接口：

```python
# 无跨轮缓存：每轮重算整个前缀
output = attention(q_all, k_all, v_all)

# 本轮只有当前 token 的 Q/K/V，历史由运行时保存
output = decode_attention(q, new_k, new_v, layer_id, batch, pool, req_to_token)
```

填写伪代码的空白，不用 PyTorch。`pool.k/v` 用 `[layer][slot]` 表示位置，`attend` 已实现当前 Q 对提供的 K/V 做 attention。

```python
def decode_attention(q, new_k, new_v, layer_id, batch, pool, req_to_token):
    for i in range(len(q)):
        slot = ____
        pool.k[____][____] = new_k[i]
        pool.v[____][____] = new_v[i]

    outputs = []
    for i in range(len(q)):
        row = ____
        length = ____
        slots = ____
        keys = [____ for slot in slots]
        values = [____ for slot in slots]
        outputs.append(attend(q[i], keys, values))
    return outputs
```

解释：为什么只有 `out_cache_loc` 不够？为什么 runtime 管理 pool 后不必每轮返回一份完整 `new_cache`？模型权重是否因此需要重新训练？

过关：先写后读，读到各请求自己的历史与当前 KV，说明模型计算和缓存管理的职责。

## Drill 4 · 用可手算的 attention 抓错误

采用单层、单 head、head_dim=1 的教学模型。直接给定 Q/K/V，不涉及 embedding、投影或 RoPE；比例因子为 1。

- A 的历史 K 都是 0，历史 V 为 `[2,4,6]`；当前 Q=1、K=0、V=8。
- B 的历史 K 都是 0，历史 V 为 `[10,20]`；当前 Q=1、K=0、V=30。
- attention 是 `softmax(QKᵀ) V`；缓存未写的新 slot 预填 K=0、V=100，作为错误哨兵。

1. 正确写入后，A、B 的 attention 输出分别是什么？
2. 漏掉当前 token：只读历史时，输出分别是什么？
3. 索引包含新 slot，却漏掉 KV 写入时，输出分别是什么？
4. 若错误地让两个请求都读取整个 batch 的所有 7 个有效 slots，会得到什么？

提示：所有 attention score 相同，因此权重均匀。真实未初始化缓存不保证是 0 或 100；这里专门指定哨兵让测试确定可复现。

过关：四种行为有不同结果，并能解释为何请求隔离属于正确性要求。

## Drill 5 · 相同 slot，为什么不同层不覆盖？

两个 attention 层都处理 A 的当前 token，使用 slot 101：

```text
layer 0: K=0.1, V=1.0
layer 1: K=0.2, V=2.0
```

1. 用字典或二维表写出正确存储位置。
2. 如果缓存只用 `pool[slot]` 做 key，结果是什么？写一个会失败的断言。
3. 模型的 `layer_id` 在哪里建立？写入接口又在哪里使用它？

源码核对：`LlamaAttention.__init__` 的 `RadixAttention(...)`；`MHATokenToKVPool.set_kv_buffer`。只看普通路径，注意 override、pipeline layer offset 等可能改变真实索引，不能把教学布局当作所有后端的物理布局。

过关：理解寻址同时需要层与 slot，能指出两个真实代码位置。

## Drill 6 · 独立追踪真实调用链

前五题完成后做，不先看 solutions 的源码导航。以 `LlamaAttention.forward` 为唯一入口，回答：

1. 当前 Q/K/V 在哪里产生？哪个调用把它们交给 SGLang 的 attention 接口？
2. `self.attn` 是什么类型？普通 eager 路径怎样进入所选 backend？
3. 假定选用 Triton backend，哪里决定 KV write location？哪里写入？
4. 历史读取使用哪些 metadata？为何与 write location 不同？
5. `ForwardBatch` 的核心字段在哪里从 `ScheduleBatch` 传入？这种赋值是否自动复制 GPU tensor？

交付一张 5–7 个框的调用图，每条边标一个真实参数，并为每一项结论记录文件、符号和当前行号。可使用 `rg -n '符号' python/sglang/srt`；不要通读整个大文件。

最后写一段设计判断：这套接口避免了什么重复工作、增加了什么 metadata 与状态约束、何种证据能揭示请求错配？“一定更快”不是静态阅读可以证明的结论。

过关：独立找到至少三个关键调用点，明确静态证据与尚未运行验证的边界。

## 可选 Drill 7 · 先估算，再讨论值不值得

自设普通 dense KV：32 层、8 个 KV heads、head_dim=128、每元素 2 bytes，每请求缓存 4096 tokens，8 个请求，无 prefix 共享、压缩或分片。

1. 逻辑 KV 总容量是多少？
2. 如果仅为改变 batch 顺序就整体搬动这些 KV，若这次搬运需完整读一遍并写一遍，HBM 总流量是多少？
3. 为什么重排少量 metadata 能避免上述大搬运，但也不能说 metadata 构建“零成本”？

容量估算不包含 padding、预留空闲 slots、临时张量或 allocator metadata；不要把估算记作实测。
