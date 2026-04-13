"""
对 workload.sql 中涉及的所有 2-way join 对（不加 filter、全表），
比较 DP partition 与 Uniform partition 的分桶效果。

去重后的基础表+key 组合（6 个）各做一次 partition 对比；
7 对 2-way join result 各做一次 partition 对比。

值域：account_id → D=15000, client_id → D=14000
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


# ── 统计 ─────────────────────────────────────────────────────────────────────

def _bucket_real_counts(buckets) -> list[int]:
    return [sum(f for k, f, _ in b if k != DUMMY_KEY) for b in buckets]


def _bucket_total_counts(buckets) -> list[int]:
    out = []
    for b in buckets:
        real_rows   = sum(f for k, f, _ in b if k != DUMMY_KEY)
        real_tuples = sum(1 for k, _, _ in b if k != DUMMY_KEY)
        out.append(real_rows + (len(b) - real_tuples))
    return out


def _cv(xs: list[int]) -> float:
    a = np.asarray(xs, dtype=float)
    if a.size == 0 or a.mean() <= 0:
        return float("nan")
    return float(a.std() / a.mean())


# ── Meta ─────────────────────────────────────────────────────────────────────

def _get_meta(keys: list, eps: float, delta: float) -> JoinMetadata:
    if not keys:
        return JoinMetadata(a=1, b=1)
    true_mf = max(Counter(keys).values())
    if true_mf == 1:
        return JoinMetadata(a=1, b=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        return JoinMetadata.from_base_table(keys, eps, delta)


# ── Partition / Bucket (quiet) ───────────────────────────────────────────────

def _dp_partition(sorted_stats, D, eps, delta, sensitivity):
    buf = io.StringIO()
    with redirect_stdout(buf):
        part = PrivatePartition(epsilon=eps, delta=delta, domain_size=D, sensitivity=sensitivity)
        partitions = part.run_partition(sorted_stats)
        buck = BucketProcessor(partitions=partitions, epsilon=eps, delta=delta, sensitivity=sensitivity)
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return partitions, buckets


def _uni_partition(sorted_stats, uni_parts, eps, delta, sensitivity):
    buf = io.StringIO()
    with redirect_stdout(buf):
        buck = BucketProcessor(partitions=uni_parts, epsilon=eps, delta=delta, sensitivity=sensitivity)
        buckets = buck.distribute_and_pad(sorted_stats, dummy_key=DUMMY_KEY)
    return buckets


# ── Hash Join (2-way) ────────────────────────────────────────────────────────

def hash_join_2way(records_A, records_B, key: str) -> list[tuple[int, int]]:
    freq_A = Counter(r[key] for r in records_A)
    freq_B = Counter(r[key] for r in records_B)
    common = set(freq_A) & set(freq_B)
    return sorted((k, freq_A[k] * freq_B[k]) for k in common)


# ── 打印 ─────────────────────────────────────────────────────────────────────

def _print_summary(tag, method, real, total):
    n = len(real)
    empty = sum(1 for x in real if x == 0)
    print(f"  [{tag}] {method}")
    print(f"    桶数={n}, 空桶={empty}, "
          f"real: min={min(real) if n else 0} max={max(real) if n else 0} "
          f"mean={np.mean(real) if n else 0:.1f} CV={_cv(real):.4f}")
    print(f"    total(real+dummy): mean={np.mean(total) if n else 0:.1f} CV={_cv(total):.4f}")


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _stats_str(real: list[int]) -> str:
    n = len(real)
    if n == 0:
        return "n=0"
    return (f"n={n}, max={max(real):,}, "
            f"mean={np.mean(real):,.0f}, CV={_cv(real):.3f}")


def _plot_domain_bar(ax, partitions, real_counts, color, title, D):
    for (start, end), count in zip(partitions, real_counts):
        ax.bar(start, count, width=(end - start + 1),
               align="edge", color=color, alpha=0.80,
               edgecolor="white", linewidth=0.4)
    # 空桶标记：在底部画红色细线，确保分区覆盖到 D 可见
    for (start, end), count in zip(partitions, real_counts):
        if count == 0:
            ax.plot([start, end], [0, 0], color="red", linewidth=2.5, solid_capstyle="butt")
    ax.set_xlim(0, D)
    ax.set_xlabel("Domain value")
    ax.set_ylabel("Real row count")
    ax.set_title(title, fontsize=10)


def _overlay_line(ax, partitions, real_counts, D, color, label, marker):
    """在值域横轴上，每个桶的区间中点处画一个点，连线。count=0 的桶用 0.5 替代以便在 log 轴上可见。"""
    xs = [0.5 * (s + e) for s, e in partitions]
    ys = [c if c > 0 else 0.5 for c in real_counts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=3,
            linewidth=1.2, alpha=0.85, label=label)
    # 空桶处额外标一个红色 × 号
    for x, c in zip(xs, real_counts):
        if c == 0:
            ax.plot(x, 0.5, marker="x", color="red", markersize=7, zorder=5)


def plot_comparison(
    label: str,
    parts_dp, dp_real, parts_uni, uni_real,
    D: int, out_dir: str,
):
    """
    3 行布局:
      行 0: 值域条形图（左=DP, 右=Uniform），各自独立 y 轴
      行 1: Overlay 折线图（两条线叠加，对数纵轴）
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Partition comparison — {label}", fontsize=14, y=0.98)

    ax_dp  = fig.add_subplot(2, 2, 1)
    ax_uni = fig.add_subplot(2, 2, 2)
    ax_line = fig.add_subplot(2, 1, 2)

    # 行 0：值域条形图（独立 y 轴）
    _plot_domain_bar(ax_dp, parts_dp, dp_real, "steelblue",
                     f"DP partition  ({_stats_str(dp_real)})", D)
    _plot_domain_bar(ax_uni, parts_uni, uni_real, "darkorange",
                     f"Uniform split  ({_stats_str(uni_real)})", D)

    # 行 1：Overlay 折线图（对数纵轴）
    _overlay_line(ax_line, parts_dp, dp_real, D,
                  "steelblue", f"DP  ({_stats_str(dp_real)})", "o")
    _overlay_line(ax_line, parts_uni, uni_real, D,
                  "darkorange", f"Uniform  ({_stats_str(uni_real)})", "s")
    ax_line.set_yscale("log")
    ax_line.set_xlim(0, D)
    ax_line.set_xlabel("Domain value")
    ax_line.set_ylabel("Real row count (log)")
    ax_line.set_title("Overlay: DP vs Uniform (log scale)")
    ax_line.legend(fontsize=9)
    ax_line.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    fig.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=150)
    plt.close(fig)

    # 单独导出 overlay 折线图（PPT 友好）
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    _overlay_line(ax2, parts_dp, dp_real, D,
                  "steelblue", f"DP  ({_stats_str(dp_real)})", "o")
    _overlay_line(ax2, parts_uni, uni_real, D,
                  "darkorange", f"Uniform  ({_stats_str(uni_real)})", "s")
    ax2.set_yscale("log")
    ax2.set_xlim(0, D)
    ax2.set_xlabel("Domain value", fontsize=12)
    ax2.set_ylabel("Real row count (log)", fontsize=12)
    ax2.set_title(f"{label}: DP vs Uniform (log scale)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{safe}_overlay.png"), dpi=200)
    plt.close(fig2)


def plot_cv_summary(rows: list[tuple[str, str, float]], out_dir: str):
    if not rows:
        return
    labels = list(dict.fromkeys(r[0] for r in rows))
    x = np.arange(len(labels))
    width = 0.35
    cv_dp  = [next((r[2] for r in rows if r[0] == t and r[1] == "DP"),      np.nan) for t in labels]
    cv_uni = [next((r[2] for r in rows if r[0] == t and r[1] == "Uniform"), np.nan) for t in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    ax.bar(x - width / 2, cv_dp,  width, label="DP",      color="steelblue")
    ax.bar(x + width / 2, cv_uni, width, label="Uniform", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CV (std / mean)")
    ax.set_title("Real-load uniformity — lower CV is more uniform")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cv_summary.png"), dpi=150)
    plt.close(fig)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    EPS   = 1.5
    DELTA = 1e-5
    UNIFORM_BINNUM = 8
    D_ACCOUNT_ID = 15000
    D_CLIENT_ID  = 14000
    BASE_PATH    = "./financial/"
    OUT_DIR = os.path.join(os.path.dirname(__file__) or ".", "comp_par_figures")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 加载 5 张表（全表，无 filter）──────────────────────────────────────
    print("加载数据（全表，无 filter）...")
    buf = io.StringIO()
    with redirect_stdout(buf):
        t_account = load_real_table(BASE_PATH + "account.csv", "Account", "account_id")
        t_trans   = load_real_table(BASE_PATH + "trans.csv",   "Trans",   "account_id")
        t_order   = load_real_table(BASE_PATH + "order.csv",   "Order",   "account_id")
        t_disp    = load_real_table(BASE_PATH + "disp.csv",    "Disp",    "account_id")
        t_client  = load_real_table(BASE_PATH + "client.csv",  "Client",  "client_id")

    tables = {
        "Account": t_account, "Trans": t_trans, "Order": t_order,
        "Disp": t_disp, "Client": t_client,
    }
    for name, t in tables.items():
        if t is None or len(t.payloads) == 0:
            print(f"  {name} 加载失败或为空，退出。")
            sys.exit(1)
        print(f"  {name}: {len(t.payloads)} 行")

    # disp 表需要同时支持 client_id 和 account_id 两种 key，
    # 但 load_real_table 只设置了一个 join_key_name。
    # 这里我们用 payloads（完整字典列表）直接操作，不依赖 Table.keys。

    uni_parts_aid = generate_uniform_partitions(D_ACCOUNT_ID, UNIFORM_BINNUM)
    uni_parts_cid = generate_uniform_partitions(D_CLIENT_ID,  UNIFORM_BINNUM)

    print(f"\n值域: account_id → D={D_ACCOUNT_ID}, client_id → D={D_CLIENT_ID}")
    print(f"Uniform binnum={UNIFORM_BINNUM}, 每步 eps={EPS}, delta={DELTA}")
    print(f"输出目录: {os.path.abspath(OUT_DIR)}\n")

    cv_rows: list[tuple[str, str, float]] = []

    # ── 去重的基础表+key 组合 ─────────────────────────────────────────────
    # (表名, key名, D, records, uniform_parts)
    base_table_key_combos: list[tuple[str, str, int, list, list]] = [
        ("Account",      "account_id", D_ACCOUNT_ID, t_account.payloads, uni_parts_aid),
        ("Trans",        "account_id", D_ACCOUNT_ID, t_trans.payloads,   uni_parts_aid),
        ("Order",        "account_id", D_ACCOUNT_ID, t_order.payloads,   uni_parts_aid),
        ("Disp_aid",     "account_id", D_ACCOUNT_ID, t_disp.payloads,    uni_parts_aid),
        ("Client",       "client_id",  D_CLIENT_ID,  t_client.payloads,  uni_parts_cid),
        ("Disp_cid",     "client_id",  D_CLIENT_ID,  t_disp.payloads,    uni_parts_cid),
    ]

    # 缓存基础表的 meta 和 partition 结果
    base_meta:   dict[str, JoinMetadata] = {}
    base_dp_res: dict[str, tuple[list, list[int]]] = {}  # (parts_dp, dp_real)

    print("=" * 70)
    print("  阶段 1: 基础表 partition 对比（6 个去重组合）")
    print("=" * 70)

    for tag, key, D, records, uni_parts in base_table_key_combos:
        keys = [r[key] for r in records]
        meta = _get_meta(keys, EPS, DELTA)
        base_meta[tag] = meta

        sorted_stats = preprocess_table_data(records, key)

        parts_dp, bucks_dp = _dp_partition(sorted_stats, D, EPS, DELTA, meta.b)
        bucks_uni = _uni_partition(sorted_stats, uni_parts, EPS, DELTA, meta.b)

        dp_real   = _bucket_real_counts(bucks_dp)
        uni_real  = _bucket_real_counts(bucks_uni)
        dp_total  = _bucket_total_counts(bucks_dp)
        uni_total = _bucket_total_counts(bucks_uni)

        base_dp_res[tag] = (parts_dp, dp_real)

        print(f"\n[{tag}] key={key}, D={D}, a={meta.a}, b={meta.b}, rows={len(records)}")
        _print_summary(tag, "DP", dp_real, dp_total)
        _print_summary(tag, "Uniform", uni_real, uni_total)

        plot_comparison(f"base_{tag}", parts_dp, dp_real, uni_parts, uni_real, D, OUT_DIR)
        cv_rows.append((tag, "DP",      _cv(dp_real)))
        cv_rows.append((tag, "Uniform", _cv(uni_real)))

    # ── 7 对 2-way join ──────────────────────────────────────────────────
    join_pairs = [
        # (左表tag, 右表tag, 左records, 右records, key, D, uni_parts)
        ("Client",   "Disp_cid",  t_client.payloads, t_disp.payloads, "client_id",  D_CLIENT_ID,  uni_parts_cid),
        ("Account",  "Trans",     t_account.payloads, t_trans.payloads, "account_id", D_ACCOUNT_ID, uni_parts_aid),
        ("Account",  "Order",     t_account.payloads, t_order.payloads, "account_id", D_ACCOUNT_ID, uni_parts_aid),
        ("Account",  "Disp_aid",  t_account.payloads, t_disp.payloads, "account_id", D_ACCOUNT_ID, uni_parts_aid),
        ("Order",    "Trans",     t_order.payloads, t_trans.payloads,   "account_id", D_ACCOUNT_ID, uni_parts_aid),
        ("Trans",    "Disp_aid",  t_trans.payloads, t_disp.payloads,    "account_id", D_ACCOUNT_ID, uni_parts_aid),
        ("Order",    "Disp_aid",  t_order.payloads, t_disp.payloads,    "account_id", D_ACCOUNT_ID, uni_parts_aid),
    ]

    print(f"\n{'=' * 70}")
    print("  阶段 2: 2-way join result partition 对比（7 对）")
    print("=" * 70)

    for tag_L, tag_R, rec_L, rec_R, key, D, uni_parts in join_pairs:
        jlabel = f"{tag_L} ⋈ {tag_R}"
        join_freq = hash_join_2way(rec_L, rec_R, key)
        if not join_freq:
            print(f"\n[{jlabel}] 无交集 key，跳过。")
            continue

        total_rows = sum(f for _, f in join_freq)
        true_max   = max(f for _, f in join_freq)

        meta_L = base_meta[tag_L]
        meta_R = base_meta[tag_R]
        meta_J = meta_L.join(meta_R)
        sens   = meta_J.b

        sorted_stats = [(k, f, []) for k, f in join_freq]
        parts_dp, bucks_dp = _dp_partition(sorted_stats, D, EPS, DELTA, sens)
        bucks_uni = _uni_partition(sorted_stats, uni_parts, EPS, DELTA, sens)

        dp_real   = _bucket_real_counts(bucks_dp)
        uni_real  = _bucket_real_counts(bucks_uni)
        dp_total  = _bucket_total_counts(bucks_dp)
        uni_total = _bucket_total_counts(bucks_uni)

        print(f"\n[{jlabel}] key={key}, D={D}")
        print(f"  {len(join_freq)} 个 key, {total_rows:,} 行, true_max_freq={true_max}")
        print(f"  传播链: a_L={meta_L.a} × a_R={meta_R.a} → sensitivity={sens}")
        _print_summary(jlabel, "DP", dp_real, dp_total)
        _print_summary(jlabel, "Uniform", uni_real, uni_total)

        plot_comparison(f"join_{tag_L}_{tag_R}", parts_dp, dp_real, uni_parts, uni_real, D, OUT_DIR)
        cv_rows.append((jlabel, "DP",      _cv(dp_real)))
        cv_rows.append((jlabel, "Uniform", _cv(uni_real)))

    # ── CV 汇总 ──────────────────────────────────────────────────────────
    plot_cv_summary(cv_rows, OUT_DIR)
    print(f"\n全部完成。图像已保存至 {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
