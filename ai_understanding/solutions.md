# 参考答案：先独立做题，再展开

这些是教学场景的答案。真实代码位置以符号为准，仓库更新后行号可能变化。

<details>
<summary>Drill 1：字段与时间边界</summary>

```text
input_ids        = [42,55]
positions        = [3,2]
out_cache_loc    = [101,205]
req_pool_indices = [7,9]
seq_lens         = [4,3]
```

五个向量 shape 都为 `[2]`。准备阶段包含新 token 的长度与位置，但 KV 尚待模型计算后写入。token ID 是词表索引；position 是序列位置；row 是请求映射表行号；slot 是 KV 存储位置编号。

每层本轮生成两组 token K/V，32 层共 64 组；不是只算第一层，然后所有层共享其 K/V。

源码：`ScheduleBatch.prepare_for_decode` 调用 `alloc_for_decode` 后递增长度；`ForwardBatch.init_new` 的普通 decode 分支调用 `clamp_position`。其 native 实现为 `clamp(seq_lens - 1, min=0)`。

</details>

<details>
<summary>Drill 2：请求对应关系</summary>

错误实现的下标 0：token 55 被当成 A，position=3，写入 101，读取 A 的 row 7。

正确重排：

```text
input_ids        = [55,42]
positions        = [2,3]
out_cache_loc    = [205,101]
req_pool_indices = [9,7]
seq_lens         = [3,4]
```

row 7/9 内容与历史 KV 保持原地。改变的是 batch 遍历顺序。如果已有按 batch 排列的 backend metadata，也要重建或相应重排，不能只考虑这五个字段。

测试用请求身份做独立 oracle，例如期待 `row→(token,position,write_slot,length)` 为 `{7:(42,3,101,4),9:(55,2,205,3)}`，重排前后都应保持该映射。只检查 shape 捕获不了错配；预期映射不能从已经错配的向量反推生成。

B 退出后，五个向量依次是 `[42]`、`[3]`、`[101]`、`[7]`、`[4]`。

</details>

<details>
<summary>Drill 3：最小伪代码</summary>

```python
def decode_attention(q, new_k, new_v, layer_id, batch, pool, req_to_token):
    for i in range(len(q)):
        slot = batch.out_cache_loc[i]
        pool.k[layer_id][slot] = new_k[i]
        pool.v[layer_id][slot] = new_v[i]

    outputs = []
    for i in range(len(q)):
        row = batch.req_pool_indices[i]
        length = batch.seq_lens[i]
        slots = req_to_token[row][:length]
        keys = [pool.k[layer_id][slot] for slot in slots]
        values = [pool.v[layer_id][slot] for slot in slots]
        outputs.append(attend(q[i], keys, values))
    return outputs
```

`out_cache_loc` 只列本轮新写位置，没有完整历史。运行时持有并更新缓存，所以模型无需返回完整新 cache 来传递其所有权。接入缓存接口不要求重训权重；模型的执行实现需要正确适配。

真实后端常通过 kernel 并行执行；这个 Python 循环不代表 SGLang 的性能实现。

</details>

<details>
<summary>Drill 4：数值反例</summary>

所有 score 为 0，softmax 是均匀权重。

| 情形 | A 输出 | B 输出 |
| --- | --- | --- |
| 正确 | (2+4+6+8)/4 = 5 | (10+20+30)/3 = 20 |
| 漏掉当前 token | (2+4+6)/3 = 4 | (10+20)/2 = 15 |
| 漏写新 slot，读到哨兵 | (2+4+6+100)/4 = 28 | (10+20+100)/3 = 130/3 |
| 错读全部 7 个正确写入的 slots | 80/7 | 80/7 |

每种错误都可能保持合法 shape，却改变语义。真实模型的 Q 不全相同，错配时不一定产生两个相同输出；这里专门构造可手算反例。

</details>

<details>
<summary>Drill 5：层与 slot</summary>

```python
pool[(0, 101)] = (0.1, 1.0)
pool[(1, 101)] = (0.2, 2.0)
assert pool[(0, 101)] == (0.1, 1.0)
assert pool[(1, 101)] == (0.2, 2.0)
```

若只以 101 为 key，第二层会覆盖第一层。测试应先写两层，再分别读取，不能只检查最后写入层。

源码中 `LlamaAttention` 构造 `RadixAttention` 时传 `layer_id`；`MHATokenToKVPool.set_kv_buffer` 读取 `layer.layer_id` 或 override，后续再选相应 buffer。实际 pool 可按层分别持有张量，并不要求存在一个 `[所有层,所有slots,...]` 的大张量。

</details>

<details>
<summary>Drill 6：源码导航与验收边界</summary>

路径均相对仓库根目录：

| 文件 | 本题关注符号/语句 |
| --- | --- |
| `python/sglang/srt/models/llama.py` | `LlamaAttention` 构造 `RadixAttention`；`forward` 计算 Q/K/V 后调用 `self.attn(q,k,v,forward_batch)` |
| `python/sglang/srt/layers/radix_attention.py` | `RadixAttention.forward` 普通 eager 分支调用 `get_attn_backend().forward(...)` |
| `python/sglang/srt/layers/attention/triton_backend.py` | `forward_decode` 使用 `forward_batch.out_cache_loc` 构造 write location；普通 MHA 路径调用 `_set_kv_buffer`；历史索引来自 `forward_metadata.kv_indices/kv_indptr` |
| `python/sglang/srt/mem_cache/memory_pool.py` | `MHATokenToKVPool.set_kv_buffer` 使用层、位置、当前 K/V 写缓存 |
| `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker.forward_batch_generation` 构造 `ForwardBatch` 后调用 `model_runner.forward` |
| `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardBatch.init_new` 接收 `ScheduleBatch` 的核心 tensor 引用，并准备额外 metadata |

接口图可分两部分：

```text
ScheduleBatch → ForwardBatch.init_new → ModelRunner → 模型层
                                                     ↓
                     Q/K/V + ForwardBatch → RadixAttention
                                                     ↓
                 Triton backend → 当前层 KV 写入 → 历史 attention 读取
```

直接赋值 tensor 字段本身不会复制其 GPU 数据；这不能推出 init_new 的所有其他工作都没有分配或传输。历史索引如何构造可留作下一课，不必在本题展开所有后端分支。

设计判断示例：保留逐层 KV 避免每轮重算前缀；间接寻址让批次顺序变化无需搬历史 KV；代价是索引构造、缓存容量、请求对应关系与写读时序约束。用有独立预期的重排测试和 attention 数值反例检查正确性；耗时和 GPU 空泡需实际 profile。

</details>

<details>
<summary>Drill 7：容量与数据移动</summary>

每 token：`2 × 32 × 8 × 128 × 2 = 131072 bytes = 128 KiB`。

每请求 4096 tokens 为 512 MiB；8 个请求总共 4 GiB。

完整读一遍、写一遍的理想流量合计 8 GiB（4 GiB read + 4 GiB write），尚不计额外中转。改变 metadata 顺序通常不需要这种整体搬运。metadata 本身仍可能有 CPU 构造、GPU kernel、同步和传输成本，需测量；不能把这个例子的搬运当作 SGLang 每轮实际发生的工作。

</details>
