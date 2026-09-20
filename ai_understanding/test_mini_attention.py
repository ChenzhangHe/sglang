"""运行：python3 -B -m unittest discover -s ai_understanding -v"""

import math
import unittest

from mini_attention import (
    DecodeBatch, KVPool, attend, build_history_indices, decode_attention,
)


class TestAttentionMath(unittest.TestCase):
    def test_uniform_and_nonuniform_scores(self):
        self.assertAlmostEqual(attend(1, [0, 0], [2, 8]), 5)
        self.assertAlmostEqual(attend(1, [0, math.log(3)], [2, 8]), 6.5)


class TestKVPool(unittest.TestCase):
    def test_write_read_order_and_overwrite(self):
        pool = KVPool()
        pool.write(0, [101, 205], [1, 2], [8, 30])
        self.assertEqual(pool.read(0, [205, 101]), ([2, 1], [30, 8]))
        pool.write(0, [101], [3], [9])
        self.assertEqual(pool.read(0, [101, 205]), ([3, 2], [9, 30]))

    def test_layer_isolation(self):
        pool = KVPool()
        pool.write(0, [101], [0.1], [1])
        pool.write(1, [101], [0.2], [2])
        self.assertEqual(pool.read(0, [101]), ([0.1], [1]))
        self.assertEqual(pool.read(1, [101]), ([0.2], [2]))

    def test_unwritten_slot_is_not_silently_zero(self):
        with self.assertRaises(KeyError):
            KVPool().read(0, [101])


class TestHistoryIndices(unittest.TestCase):
    def test_ragged_reorder_and_unused_tail(self):
        rows = {7: [11, 18, 23, 101, 999], 9: [60, 72, 205, 888]}
        self.assertEqual(build_history_indices(rows, [7, 9], [4, 3]),
                         ([11, 18, 23, 101, 60, 72, 205], [0, 4, 7]))
        self.assertEqual(build_history_indices(rows, [9, 7], [3, 4]),
                         ([60, 72, 205, 11, 18, 23, 101], [0, 3, 7]))
        self.assertEqual(rows[7][-1], 999)

    def test_empty(self):
        self.assertEqual(build_history_indices({}, [], []), ([], [0]))


def scenario(reverse=False):
    # 直接放入已知历史，避免 fixture 依赖尚未实现的 write。
    pool = KVPool()
    for slot, value in [(11, 2), (18, 4), (23, 6), (60, 10), (72, 20),
                        (101, 100), (205, 100)]:
        pool.k[(0, slot)] = 0
        pool.v[(0, slot)] = value
    mapping = {7: [11, 18, 23, 101], 9: [60, 72, 205]}
    if reverse:
        return pool, DecodeBatch(0, [9, 7], [3, 4], [205, 101], mapping), [30, 8]
    return pool, DecodeBatch(0, [7, 9], [4, 3], [101, 205], mapping), [8, 30]


class TestDecode(unittest.TestCase):
    def test_write_before_read_and_request_isolation(self):
        pool, batch, values = scenario()
        outputs = decode_attention([1, 1], [0, 0], values, batch, pool)
        self.assertEqual(len(outputs), 2)
        for actual, expected in zip(outputs, [5, 20]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(pool.v[(0, 101)], 8)
        self.assertEqual(pool.v[(0, 205)], 30)
        self.assertEqual(pool.v[(0, 11)], 2)

    def test_reordering_preserves_each_request_result(self):
        pool, batch, values = scenario(reverse=True)
        outputs = decode_attention([1, 1], [0, 0], values, batch, pool)
        self.assertEqual(len(outputs), 2)
        for actual, expected in zip(outputs, [20, 5]):
            self.assertAlmostEqual(actual, expected)

    def test_nonuniform_attention_includes_current_token(self):
        pool = KVPool()
        pool.k[(0, 11)], pool.v[(0, 11)] = 0, 2
        batch = DecodeBatch(0, [7], [2], [101], {7: [11, 101]})
        outputs = decode_attention([1], [math.log(3)], [8], batch, pool)
        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0], 6.5)

    def test_empty(self):
        self.assertEqual(decode_attention([], [], [],
                         DecodeBatch(0, [], [], [], {}), KVPool()), [])


if __name__ == "__main__":
    unittest.main()
