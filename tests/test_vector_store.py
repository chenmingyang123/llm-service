"""W2 第 4 天 · 向量索引的单元测试。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import vector_store as vs  # noqa: E402


def make_data(n=200, dim=16, seed=0):
    """造一组语义上能分开的假向量：相邻编号互为近邻，便于验证召回。"""
    rng = np.random.default_rng(seed)
    n = n - n % 4
    base = rng.normal(size=(n // 4, dim))
    mat = np.repeat(base, 4, axis=0) + rng.normal(scale=0.05, size=(n, dim))
    rows = [{"chunk_id": f"c{i:04d}", "source": "test", "title": "t",
             "role": "normal", "parent_id": None,
             "section_path": ["s"], "vector": mat[i].astype(np.float32).tolist()}
            for i in range(n)]
    return mat.astype(np.float32), rows


def test_brute_is_exact_baseline():
    """暴力检索：召回必须恒为 1.0，它是所有对比的尺子。"""
    mat, rows = make_data()
    idx = vs.BruteForceIndex(mat, rows)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = mat[rng.integers(0, len(mat))]
        got = [i for i, _ in idx.search(q, k=5)]
        assert len(set(got)) == 5
        qn = mat[0] * 0 + q
        qn = qn / np.linalg.norm(qn)
        best = float(np.max(vs._normalize(mat) @ qn))
        assert idx.search(q, k=5)[0][1] >= best - 1e-5


def test_brute_normalizes_once_not_per_query():
    """归一化只做一次：这是 IVF 曾经踩过的坑（见 self.mat 的注释）。"""
    mat, rows = make_data()
    idx = vs.BruteForceIndex(mat, rows)
    assert idx.mat is not None and len(idx.mat) == len(rows)
    assert idx.mat is idx.mat


def test_ivf_recall_grows_with_nprobe():
    """nprobe 越大召回越高（在别的都不变的前提下）—— 这是索引的第一性原理。"""
    mat, rows = make_data(n=400)
    recs = []
    for np_ in (1, 2, 4, 8, 16):
        idx = vs.IVFIndex(mat, rows, nlist=16, nprobe=np_)
        rng = np.random.default_rng(11)
        qs = [mat[rng.integers(0, len(mat))]
              + rng.normal(scale=0.02, size=mat.shape[1])
              for _ in range(10)]
        base = vs.BruteForceIndex(mat, rows)
        truth = [[i for i, _ in base.search(q, k=5)] for q in qs]
        got = [[i for i, _ in idx.search(q, k=5)] for q in qs]
        recs.append(sum(vs.recall_at_k(g, t, 5) for g, t in zip(got, truth)) / len(qs))
    assert recs == sorted(recs), f"召回应随 nprobe 单调不减，实际 {recs}"
    assert recs[-1] > recs[0], "nprobe 拉满却没召回提升，分簇一定有问题"


def test_ivf_nlist_cannot_exceed_n():
    """簇数不能超过向量数，也不能是 0——否则索引直接失效。"""
    mat, rows = make_data(n=8)
    idx = vs.IVFIndex(mat, rows, nlist=999, nprobe=1)
    assert idx.nlist <= 10
    assert all(len(b) > 0 for b in idx.buckets), "不该出现空簇"


def test_recall_at_k_edge_cases():
    assert vs.recall_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0
    assert vs.recall_at_k([9, 2, 3], [1, 2, 3], 3) == 2 / 3
    assert vs.recall_at_k([9, 8, 3], [1, 2, 3], 3) == 1 / 3
    assert vs.recall_at_k([], [1, 2, 3], 3) == 0.0
    assert vs.recall_at_k([1], [1], 0) == 0.0


def test_perturb_changes_vector_but_stays_unit():
    """加噪声是为了让自采样查询不那么"恰好"落在原簇里。"""
    v = [0.1] * 16
    p = vs.perturb(v, sigma=0.05, seed=1)
    assert p != v
    assert abs(float(np.linalg.norm(p)) - 1.0) < 1e-5
    assert vs.perturb(v, sigma=0.0) == v


def test_load_vectors_skips_dirty_lines(tmp_path):
    """语料里有空行和行首游离字符，读档不能被它们带崩（真发生过）。"""
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        '{"chunk_id": "a", "vector": [1.0, 0.0]}\n'
        "\n"
        '{"chunk_id": "b", "vector": [0.0, 1.0]}\n'
        '吗{"chunk_id": "c", "vector": null}\n'
        "not json at all\n",
        encoding="utf-8")
    mat, rows = vs.load_vectors(str(p))
    assert [r["chunk_id"] for r in rows] == ["a", "b"], "只有带向量的才进库"
    assert len(mat) == 2 and mat.shape[1] == 2


def test_build_dispatch():
    mat, rows = make_data(n=40)
    assert isinstance(vs.build("brute", mat, rows), vs.BruteForceIndex)
    assert isinstance(vs.build("ivf", mat, rows, nlist=4, nprobe=1), vs.IVFIndex)
    with pytest.raises(ValueError):
        vs.build("不存在", mat, rows)


def test_milvus_roundtrip_real_index(tmp_path: str):
    """走真实 Milvus Lite（落临时文件），验证"建→查"链路通。"""
    pytest.importorskip("pymilvus")
    mat, rows = make_data(n=60, dim=8)
    db = str(tmp_path / "m.db")
    idx = vs.MilvusIndex(mat, rows, db_path=db)
    res = idx.search(mat[0], k=3)
    assert len(res) == 3
    ids = [i for i, _ in res]
    assert ids[0] == 0, "最近邻必须排第一"
    assert len(set(ids)) == 3


def test_milvus_search_batch_matches_single(tmp_path):
    """批量接口和单条接口必须给出同样的排序，否则批量的数没法跟单条比。"""
    pytest.importorskip("pymilvus")
    mat, rows = make_data(n=60, dim=8)
    idx = vs.MilvusIndex(mat, rows, db_path=str(tmp_path / "m2.db"))
    one = [i for i, _ in idx.search(mat[0], k=5)]
    many = [i for i, _ in idx.search_batch([mat[0]], k=5)[0]]
    assert one == many
