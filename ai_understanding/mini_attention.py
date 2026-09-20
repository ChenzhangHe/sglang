"""学习用单 head、标量 Q/K/V decode；仅需 Python 标准库。

完成四处 TODO。这里没有 token embedding、RoPE 或 GPU kernel。
每个列表的下标必须始终对应同一个请求。
"""

from dataclasses import dataclass
from math import exp


@dataclass
class DecodeBatch:
    layer_id: int
    req_pool_indices: list[int]
    seq_lens: list[int]  # 包含当前 token
    out_cache_loc: list[int]
    req_to_token: dict[int, list[int]]  # 已包含本轮新分配的 slot


class KVPool:
    def __init__(self):
        self.k: dict[tuple[int, int], float] = {}
        self.v: dict[tuple[int, int], float] = {}

    def write(self, layer_id, slots, keys, values):
        """对应位置逐个写入；key 为 (layer_id, slot)，覆盖同位置旧值。

        本题保证三个输入列表等长。不要复制或移动其他 slots。
        """
        # TODO 1
        raise NotImplementedError("完成 KVPool.write")

    def read(self, layer_id, slots):
        """返回 (keys, values)，保持 slots 的顺序；未写位置抛 KeyError。"""
        # TODO 2
        raise NotImplementedError("完成 KVPool.read")


def build_history_indices(req_to_token, rows, seq_lens):
    """返回 (kv_indices, kv_indptr)，只取各 row 的有效前缀。

    例如 rows=[7,9]、lengths=[4,3] 得到边界 [0,4,7]。
    空 batch 返回 ([], [0])。输入长度合法，无需实现参数校验。
    """
    # TODO 3
    raise NotImplementedError("完成 build_history_indices")


def attend(query, keys, values):
    """已提供：数值稳定的标量 softmax(QKᵀ)V；head_dim=1。"""
    if not keys or len(keys) != len(values):
        raise ValueError("需要非空且等长的 K/V")
    scores = [query * key for key in keys]
    maximum = max(scores)
    weights = [exp(score - maximum) for score in scores]
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def decode_attention(queries, new_keys, new_values, batch, pool):
    """返回每请求一个标量输出，并把新 KV 写入 pool。

    输入列表均按 batch 顺序排列且等长，seq_lens 均大于零。
    使用上面三个 TODO 函数和 attend：先写当前 KV，构造历史索引，
    然后按 kv_indptr 取每请求的索引片段，读缓存并计算 attention。
    空 batch 返回 []。不要修改 batch 中的映射或历史 KV。
    """
    # TODO 4
    raise NotImplementedError("完成 decode_attention")
