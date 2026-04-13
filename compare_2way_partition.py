"""
2-way Join 场景下的 DP partition vs Uniform partition 对比。

观察对象（两组独立实验）：
  实验 1: Account ⋈ Trans  (join key = account_id)
  实验 2: Account ⋈ Order  (join key = account_id)

每组实验分别展示：
  - 两张基础表各自的 partition 结果
  - 2-way join 结果的 partition 结果

输出：
  - 终端统计
  - 图像保存至 compare_2way_figures/
"""

from __future__ import annotations

import io
import os
import sys
from collections import Counter
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

from bucket_mechanism import BucketProcessor
from join_mechanism import JoinMetadata
from private_partition import PrivatePartitionOffline as PrivatePartition
from utils import (
    generate_uniform_partitions,
    load_real_table,
    preprocess_table_data,
)


DUMMY_KEY = -999


# ── 统计辅助 ─────────────────────────────────────────────────────────────────

def _bucket_real_counts(buckets) -> list[int]:
    return [sum(f for k, f, _ in bucket if k != DUMMY_KEY) for bucket in buckets]


def _bucket_total_counts(buckets) -> list[int]:
    out = []
    for bucket in buckets:
        real_rows   = sum(f for k, f, _ in bucket if k != DUMMY_KEY)
        real_tuples = sum(1 for k, _, _ in bucket if k != DUMMY_KEY)
        dummies     = len(bucket) - real_tuples
        out.append(real_rows + dummies)
    return out


def _cv(xs: list[int]) -> float:
    a = np.asarray(xs, dtype=float)
    if a.size == 0 or a.mean() <= 0:
        return float("nan")
    return float(a.std() / a.mean())


# ── JoinMetadata 辅助 ────────────────────────────────────────────────────────

def _get_meta(keys: list, eps: float, delta: float) -> JoinMetadata:
    """
    若该表在 join key 上的真实最大频次 == 1（主键），视为公开，直接 a=1。
    否则走正常 from_base_table DP 估算。
    """
    if not keys:
        return JoinMetadata(a=1, b=1)
    true_mf = max(Counter(keys).values())
    if true_mf == 1:
        return JoinMetadata(a=1, b=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        return JoinMetadata.from_base_table(keys, eps, delta)


# ── Hash Join (2-way) ─────────────────────────────────────────────────────────

def hash_join_2way(
    records_A, records_B, key: str
) -> list[tuple[int, int]]:
    """
    2-way 内存 Hash Join。
    返回 [(key, freq_A[k]*freq_B[k]), ...] 按 key 升序，仅保留两表都有的 key。
    """
    freq_A = Counter(r[key] for r in records_A)
    freq_B = Counter(r[key] for r in records_B)
    common = set(freq_A) & set(freq_B)
    return sorted((k, freq_A[k] * freq_B[k]) for k in common)


def _build_sorted_stats(freq_pairs: list[tuple[int, int]]) -> list[tuple[int, int, list]]:
    return [(k, f, []) for k, f in freq_pairs]


# ── 分区 / 装桶（静默版） ──────────────────────────────────────────────────────

def run_partition_and_bucket_quiet(
    records, join_key: str, D: int,
    eps_part: float, delta_part: float,
    eps_buck: float, delta_buck: float,
    sensitivity: int,
) -> tuple[list, list]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        sorted_stats = preprocess_table_data(records, join_key)
        part = PrivatePartition(
            epsilon=eps_part, delta=delta_part,
            domain_size=D, sensitivity=sensitivity,
        )
        partitions = part.run_partition(sorted_stats)
        buck = BucketProcessor(
            partitions=partitions,
            epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity,
        )
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return partitions, buckets


def run_partition_and_bucket_from_stats_quiet(
    sorted_stats: list, D: int,
    eps_part: float, delta_part: float,
    eps_buck: float, delta_buck: float,
    sensitivity: int,
) -> tuple[list, list]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        part = PrivatePartition(
            epsilon=eps_part, delta=delta_part,
            domain_size=D, sensitivity=sensitivity,
        )
        partitions = part.run_partition(sorted_stats)
        buck = BucketProcessor(
            partitions=partitions,
            epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity,
        )
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return partitions, buckets


def run_uniform_bucket_quiet(
    records, join_key: str, D: int,
    partitions: list,
    eps_buck: float, delta_buck: float,
    sensitivity: int,
) -> list:
    buf = io.StringIO()
    with redirect_stdout(buf):
        sorted_stats = preprocess_table_data(records, join_key)
        buck = BucketProcessor(
            partitions=partitions,
            epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity,
        )
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return buckets


def run_uniform_bucket_from_stats_quiet(
    sorted_stats: list, D: int,
    partitions: list,
    eps_buck: float, delta_buck: float,
    sensitivity: int,
) -> list:
    buf = io.StringIO()
    with redirect_stdout(buf):
        buck = BucketProcessor(
            partitions=partitions,
            epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity,
        )
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return buckets


# ── 打印汇总 ──────────────────────────────────────────────────────────────────

def summarize(name: str, method: str, real_counts: list[int], total_counts: list[int]):
    n     = len(real_counts)
    empty = sum(1 for x in real_counts if x == 0)
    print(f"[{name}] {method}")
    print(f"  桶数: {n}  |  空桶(实载为0): {empty}")
    print(
        f"  实载(行/桶): min={min(real_counts) if n else 0}, "
        f"max={max(real_counts) if n else 0}, "
        f"mean={np.mean(real_counts) if n else 0:.2f}, "
        f"CV={_cv(real_counts):.4f}"
    )
    print(
        f"  总物理项(实+Dummy/桶): min={min(total_counts) if n else 0}, "
        f"max={max(total_counts) if n else 0}, "
        f"mean={np.mean(total_counts) if n else 0:.2f}, "
        f"CV={_cv(total_counts):.4f}"
    )


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _plot_domain_view(ax, partitions, real_counts, color, title, D):
    for (start, end), count in zip(partitions, real_counts):
        ax.bar(
            start, count, width=(end - start + 1),
            align="edge", color=color, alpha=0.80,
            edgecolor="white", linewidth=0.4,
        )
    ax.set_xlim(0, D)
    ax.set_xlabel("Domain value")
    ax.set_ylabel("Real row count")
    ax.set_title(title)


def _plot_bucket_index_view(ax, real_counts, color, title):
    x = np.arange(len(real_counts))
    ax.bar(x, real_counts, color=color, alpha=0.85, width=0.8)
    ax.set_xlabel("Bucket index")
    ax.set_ylabel("Real row count")
    ax.set_title(title)


def plot_table_comparison(
    table_name, parts_dp, dp_real, parts_uni, uni_real, D, out_dir,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"Partition comparison — {table_name}", fontsize=14)
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    _plot_domain_view(
        ax00, parts_dp, dp_real, "steelblue",
        f"DP partition: domain view  (n_buckets={len(parts_dp)})", D,
    )
    _plot_domain_view(
        ax01, parts_uni, uni_real, "darkorange",
        f"Uniform split: domain view  (n_buckets={len(parts_uni)})", D,
    )
    y_max_domain = max(max(dp_real, default=0), max(uni_real, default=0)) * 1.10
    ax00.set_ylim(0, y_max_domain)
    ax01.set_ylim(0, y_max_domain)

    _plot_bucket_index_view(
        ax10, dp_real, "steelblue",
        f"DP partition: count per bucket  (CV={_cv(dp_real):.3f})",
    )
    _plot_bucket_index_view(
        ax11, uni_real, "darkorange",
        f"Uniform split: count per bucket  (CV={_cv(uni_real):.3f})",
    )
    y_max_bucket = max(max(dp_real, default=0), max(uni_real, default=0)) * 1.10
    ax10.set_ylim(0, y_max_bucket)
    ax11.set_ylim(0, y_max_bucket)

    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in table_name)
    fig.savefig(os.path.join(out_dir, f"compare_{safe}.png"), dpi=150)
    plt.close(fig)


def plot_cv_summary(rows: list[tuple[str, str, float]], out_dir: str, title_suffix: str = ""):
    if not rows:
        return
    labels = list(dict.fromkeys(r[0] for r in rows))
    x = np.arange(len(labels))
    width = 0.35
    cv_dp  = [next((r[2] for r in rows if r[0] == t and r[1] == "DP"),      np.nan) for t in labels]
    cv_uni = [next((r[2] for r in rows if r[0] == t and r[1] == "Uniform"), np.nan) for t in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, cv_dp,  width, label="DP",      color="steelblue")
    ax.bar(x + width / 2, cv_uni, width, label="Uniform", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CV of real load per bucket (std / mean)")
    ax.set_title(f"Real-load uniformity — lower CV is more uniform{title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title_suffix) or "all"
    fig.savefig(os.path.join(out_dir, f"compare_cv_{safe}.png"), dpi=150)
    plt.close(fig)


# ── 单组 2-way 实验 ──────────────────────────────────────────────────────────

def run_experiment(
    name_left: str,  table_left,
    name_right: str, table_right,
    D: int, EPS: float, DELTA: float,
    uniform_parts: list,
    out_dir: str,
) -> list[tuple[str, str, float]]:
    """
    对两张表分别做 partition 对比，再对它们的 2-way join 结果做 partition 对比。
    返回 cv_rows 以便汇总。
    """
    join_label = f"{name_left} ⋈ {name_right}"
    print(f"\n{'='*70}")
    print(f"  实验: {join_label}")
    print(f"{'='*70}")

    cv_rows: list[tuple[str, str, float]] = []

    # 两张基础表的 meta
    meta_L = _get_meta([r["account_id"] for r in table_left.payloads],  EPS, DELTA)
    meta_R = _get_meta([r["account_id"] for r in table_right.payloads], EPS, DELTA)
    print(f"  {name_left}:  a={meta_L.a}, b={meta_L.b} (true_mf={'=1, public' if meta_L.a == 1 else '>' + str(1)})")
    print(f"  {name_right}: a={meta_R.a}, b={meta_R.b}")

    # 基础表 partition
    for name, table, meta in [(name_left, table_left, meta_L), (name_right, table_right, meta_R)]:
        tag = f"{join_label}/{name}"
        sens = meta.b

        parts_dp, bucks_dp = run_partition_and_bucket_quiet(
            table.payloads, "account_id", D,
            EPS, DELTA, EPS, DELTA, sens,
        )
        bucks_uni = run_uniform_bucket_quiet(
            table.payloads, "account_id", D,
            uniform_parts, EPS, DELTA, sens,
        )
        dp_real   = _bucket_real_counts(bucks_dp)
        uni_real  = _bucket_real_counts(bucks_uni)
        dp_total  = _bucket_total_counts(bucks_dp)
        uni_total = _bucket_total_counts(bucks_uni)

        summarize(tag, "DP", dp_real, dp_total)
        summarize(tag, "Uniform", uni_real, uni_total)
        print(f"  DP 分区区间数: {len(parts_dp)}\n")

        plot_table_comparison(
            f"{join_label} - {name}",
            parts_dp, dp_real, uniform_parts, uni_real, D, out_dir,
        )
        cv_rows.append((name, "DP",      _cv(dp_real)))
        cv_rows.append((name, "Uniform", _cv(uni_real)))

    # 2-way join 结果
    join_freq = hash_join_2way(table_left.payloads, table_right.payloads, "account_id")
    total_join_rows = sum(f for _, f in join_freq)
    true_max_join = max((f for _, f in join_freq), default=0)
    print(
        f"[{join_label}] {len(join_freq)} 个不同 key，"
        f"共 {total_join_rows:,} 行，true_max_freq={true_max_join}"
    )

    join_sorted_stats = _build_sorted_stats(join_freq)

    meta_join = meta_L.join(meta_R)
    sens_join = meta_join.b
    print(f"  传播链: a_L={meta_L.a} × a_R={meta_R.a} → sensitivity={sens_join}")

    parts_dp_j, bucks_dp_j = run_partition_and_bucket_from_stats_quiet(
        join_sorted_stats, D,
        EPS, DELTA, EPS, DELTA, sens_join,
    )
    bucks_uni_j = run_uniform_bucket_from_stats_quiet(
        join_sorted_stats, D,
        uniform_parts, EPS, DELTA, sens_join,
    )

    dp_real_j   = _bucket_real_counts(bucks_dp_j)
    uni_real_j  = _bucket_real_counts(bucks_uni_j)
    dp_total_j  = _bucket_total_counts(bucks_dp_j)
    uni_total_j = _bucket_total_counts(bucks_uni_j)

    jtag = f"Join({join_label})"
    summarize(jtag, "DP", dp_real_j, dp_total_j)
    summarize(jtag, "Uniform", uni_real_j, uni_total_j)
    print(f"  DP 分区区间数: {len(parts_dp_j)}\n")

    plot_table_comparison(
        jtag,
        parts_dp_j, dp_real_j, uniform_parts, uni_real_j, D, out_dir,
    )
    cv_rows.append((jtag, "DP",      _cv(dp_real_j)))
    cv_rows.append((jtag, "Uniform", _cv(uni_real_j)))

    return cv_rows


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    DOMAIN_SIZE    = 15000
    BASE_PATH      = "./financial/"
    UNIFORM_BINNUM = 8
    EPS   = 1.5
    DELTA = 1e-5
    OUT_DIR = os.path.join(os.path.dirname(__file__) or ".", "compare_2way_figures")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载数据 (join_key = account_id)...")
    buf = io.StringIO()
    with redirect_stdout(buf):
        t_account = load_real_table(
            csv_path=BASE_PATH + "account.csv",
            table_name="Account",
            join_key_name="account_id",
        )
        t_trans = load_real_table(
            csv_path=BASE_PATH + "trans.csv",
            table_name="Trans",
            join_key_name="account_id",
            filter_attribute="operation",
            filter_value="VYBER KARTOU",
        )
        t_order = load_real_table(
            csv_path=BASE_PATH + "order.csv",
            table_name="Order",
            join_key_name="account_id",
        )

    for name, t in [("Account", t_account), ("Trans", t_trans), ("Order", t_order)]:
        if t is None or len(t.payloads) == 0:
            print(f"{name} 加载失败或为空，退出。")
            sys.exit(1)
        print(f"  - {name}: {len(t.payloads)} 行")

    uniform_parts = generate_uniform_partitions(DOMAIN_SIZE, UNIFORM_BINNUM)
    print(f"值域 D={DOMAIN_SIZE}，均匀切分桶数 binnum={UNIFORM_BINNUM}")
    print(f"每步隐私预算: eps={EPS}, delta={DELTA}")
    print(f"图像输出目录: {os.path.abspath(OUT_DIR)}")

    all_cv: list[tuple[str, str, float]] = []

    # 实验 1: Account ⋈ Trans
    cv1 = run_experiment(
        "Account", t_account, "Trans", t_trans,
        DOMAIN_SIZE, EPS, DELTA, uniform_parts, OUT_DIR,
    )
    all_cv.extend(cv1)

    # 实验 2: Account ⋈ Order
    cv2 = run_experiment(
        "Account", t_account, "Order", t_order,
        DOMAIN_SIZE, EPS, DELTA, uniform_parts, OUT_DIR,
    )
    all_cv.extend(cv2)

    # 汇总 CV 对比（把两组实验的 join result 放一起）
    join_cv = [r for r in all_cv if r[0].startswith("Join(")]
    plot_cv_summary(join_cv, OUT_DIR, " (2-way joins)")

    print("\n全部完成。图像已保存至 compare_2way_figures/")


if __name__ == "__main__":
    main()
