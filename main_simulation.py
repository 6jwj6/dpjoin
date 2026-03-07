import numpy as np
import time
# 假设 data_structure.py 中有 Table 类
from data_structure import Table 
# 假设 private_partition.py 中是我们最新的 PrivatePartitionOffline 类
from private_partition import PrivatePartitionOffline as PrivatePartition 
# 假设 bucket_mechanism.py 中是我们最新的 BucketProcessor 类
from bucket_mechanism import BucketProcessor
# 引入最新编写的 Join 模块, 新增引入 JoinMetadata
from join_mechanism import DPJoiner, JoinMetadata

def generate_simulation_table(name: str, D: int, num_records: int, sensitivity: int = 1, num_clusters: int = 50, fixed_centers=None) -> Table:
    """
    [高性能数据生成器]
    生成符合大规模稀疏分布的 Table 对象，支持自定义敏感度。
    新增 fixed_centers 参数，用于确保多张表生成时有数据重叠，方便测试 Join。
    """
    print(f"[{name}] Generating Data... D={D:,}, N={num_records:,}, sens={sensitivity}")
    t0 = time.time()
    
    # 1. 生成热点数据 (80% 数据集中在少数 Cluster)
    records_in_clusters = int(num_records * 0.8)
    
    if fixed_centers is None:
        centers = np.random.randint(1, D, size=num_clusters)
    else:
        centers = fixed_centers
        num_clusters = len(centers)
        
    cluster_size = records_in_clusters // num_clusters
    
    # 使用 float 生成后转 int，模拟正态分布聚集
    cluster_data = np.random.normal(loc=centers[:, None], scale=D*0.0001, size=(num_clusters, cluster_size))
    cluster_data = cluster_data.flatten()
    
    # 2. 生成均匀分布数据 (20% 数据作为背景噪声)
    records_noise = num_records - len(cluster_data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    
    # 3. 合并并截断边界
    all_keys = np.concatenate([cluster_data, uniform_data])
    all_keys = np.clip(all_keys, 1, D).astype(int) # 确保 Key 在 [1, D] 范围内
    
    payloads = [None] * len(all_keys) 
    table = Table(name=name, keys=all_keys, payloads=payloads, sensitivity=sensitivity)
    
    print(f"[{name}] Done. Time: {time.time() - t0:.4f}s. MaxKey: {np.max(all_keys)}")
    return table

def print_summary(name: str, partitions, final_buckets, elapsed_time, table: Table, D: int):
    """
    打印单表详细统计报告
    """
    num_buckets = len(final_buckets)
    total_real_keys = 0
    total_dummy_items = 0
    empty_buckets = 0
    
    for bucket in final_buckets:
        real_in_bucket = sum(1 for k, f in bucket if k != -999)
        dummies_in_bucket = len(bucket) - real_in_bucket
        total_real_keys += real_in_bucket
        total_dummy_items += dummies_in_bucket
        if real_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_keys + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    
    print(f"\n" + "-"*60)
    print(f" [{name}] Partition & Bucket 处理报告")
    print(f"-"*60)
    print(f"  - 算法耗时:           {elapsed_time:.4f} 秒")
    print(f"  - 生成区间数:         {len(partitions):,}")
    print(f"  - 纯噪声桶 (Empty):   {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"  - 真实数据 (Real):    {total_real_keys:,}")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,}")
    if total_real_keys > 0:
        print(f"  - 膨胀系数 (Ratio):   {(total_real_keys + total_dummy_items)/total_real_keys:.2f}x")
    print(f"  - 平均 Bucket 大小:   {avg_bucket_size:.2f} items")

def print_join_summary(joined_partitions, joined_buckets, elapsed_time):
    """
    打印 Join 操作的统计报告
    """
    num_buckets = len(joined_buckets)
    total_real_keys = 0
    total_dummy_items = 0
    empty_buckets = 0
    
    for bucket in joined_buckets:
        real_in_bucket = sum(1 for k, f in bucket if k != -999)
        dummies_in_bucket = len(bucket) - real_in_bucket
        total_real_keys += real_in_bucket
        total_dummy_items += dummies_in_bucket
        if real_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_keys + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"      差分隐私 Join 阶段压力测试报告      ")
    print(f"="*60)
    print(f"性能指标:")
    print(f"  - Join 算法耗时:      {elapsed_time:.4f} 秒")
    print(f"-"*60)
    print(f"Join 结果:")
    print(f"  - 交叉对齐区间数:     {len(joined_partitions):,}")
    print(f"  - 纯噪声桶 (Empty):   {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*60)
    print(f"Join Bucket 填充:")
    print(f"  - 总项数 (Real+Dummy):{total_real_keys + total_dummy_items:,}")
    print(f"  - 真实连接结果 (Real):{total_real_keys:,}")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,}")
    if total_real_keys > 0:
        print(f"  - 膨胀系数 (Ratio):   {(total_real_keys + total_dummy_items)/total_real_keys:.2f}x")
    print(f"  - 平均 Bucket 大小:   {avg_bucket_size:.2f} items")
    print(f"="*60 + "\n")

def process_single_table(name, table, D, eps_part, delta_part, eps_buck, delta_buck):
    """辅助函数：执行单表的 Partition 和 Bucket 流程"""
    partition_algo = PrivatePartition(epsilon=eps_part, delta=delta_part, domain_size=D, sensitivity=table.sensitivity)
    
    t_start = time.time()
    sorted_stats = partition_algo.preprocess_data(table.keys)
    partitions = partition_algo.run_partition(sorted_stats)
    
    bucket_processor = BucketProcessor(partitions=partitions, epsilon=eps_buck, delta=delta_buck, sensitivity=table.sensitivity)
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    t_end = time.time()
    
    print_summary(name, partitions, final_buckets, t_end - t_start, table, D)
    return partitions, final_buckets

def run_join_benchmark(D, N_A, N_B, epsilon=1.0, delta=1e-6):
    """
    核心运行逻辑：模拟完整的 双表 Partition -> Bucket -> Join 流程
    """
    print(f"Starting Join Benchmark: eps={epsilon}, delta={delta}")

    # --- 1. 隐私预算分配 (多阶段组合) ---
    # 假设一个用户的数据同时存在于两张表中，我们需要将总预算分给 5 个阶段：
    # Table A (Part + Buck), Table B (Part + Buck), Join (加噪)
    eps_stage = epsilon / 5.0
    delta_stage = delta / 5.0

    # --- 2. 数据准备 ---
    # 使用相同的 centers 以确保热点重合，能够成功 Join 出数据
    shared_centers = np.random.randint(1, D, size=50)
    table_A = generate_simulation_table("Table_A", D, N_A, sensitivity=1, fixed_centers=shared_centers)
    table_B = generate_simulation_table("Table_B", D, N_B, sensitivity=1, fixed_centers=shared_centers)
    
    # --- 3. 分别处理 Table A 和 Table B ---
    print("\n>>> 处理 Table A ...")
    parts_A, buckets_A = process_single_table("Table_A", table_A, D, eps_stage, delta_stage, eps_stage, delta_stage)
    
    print("\n>>> 处理 Table B ...")
    parts_B, buckets_B = process_single_table("Table_B", table_B, D, eps_stage, delta_stage, eps_stage, delta_stage)

    # --- 4. 执行 Join 流程 ---
    print("\n>>> 开始执行 DP Join ...")
    # 注意：Join 的 sensitivity 取决于业务场景（最大连接度）。这里模拟设为 1 或自行调大。
    join_sensitivity = 1 
    joiner = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=join_sensitivity)
    
    t_join_start = time.time()
    joined_parts, joined_bucks = joiner.run_join(parts_A, buckets_A, parts_B, buckets_B, dummy_key=-999)
    t_join_end = time.time()
    
    # --- 5. 打印 Join 报告 ---
    print_join_summary(joined_parts, joined_bucks, t_join_end - t_join_start)
    
    return joined_parts, joined_bucks

def run_3way_join_benchmark(D, N_A, N_B, N_C, epsilon=1.0, delta=1e-6):
    print(f"\n{'='*60}")
    print(f"Starting 3-Way Join Benchmark (Non-uniform Parallel Budget): eps={epsilon}, delta={delta}")
    print(f"{'='*60}")

    # ---------------------------------------------------
    # 【终极优化】：非均匀隐私预算分配 (Non-uniform Allocation)
    # ---------------------------------------------------
    # 整个生命周期有 5 个串行处理节点：Meta, Partition, Bucket, Join1, Join2
    # 我们将最重的预算 (50%) 砸给 Meta 阶段，压制乘法爆炸的源头。
    # 剩下的 50% 由另外 4 个阶段平分 (每个 12.5%)。
    
    ratio_meta = 0.90
    ratio_others = (1.0 - ratio_meta) / 4.0

    eps_meta = epsilon * ratio_meta
    delta_meta = delta * ratio_meta

    eps_stage = epsilon * ratio_others
    delta_stage = delta * ratio_others

    print(f"[Budget Allocation] Meta Stage    : eps={eps_meta:.3f}, delta={delta_meta:.2e} ({ratio_meta*100}%)")
    print(f"[Budget Allocation] Other 4 Stages: eps={eps_stage:.3f}, delta={delta_stage:.2e} ({ratio_others*100}% each)")

    shared_centers = np.random.randint(1, D, size=50)
    table_A = generate_simulation_table("Table_A", D, N_A, fixed_centers=shared_centers)
    table_B = generate_simulation_table("Table_B", D, N_B, fixed_centers=shared_centers)
    table_C = generate_simulation_table("Table_C", D, N_C, fixed_centers=shared_centers)
    
    # ---------------------------------------------------
    # [Phase 0] 计算基础表元数据 (使用重仓预算 eps_meta)
    # ---------------------------------------------------
    print("\n[Phase 0] 计算基础表元数据 (Noisy Max Frequency)...")
    meta_A = JoinMetadata.from_base_table(table_A.keys, eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table(table_B.keys, eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table(table_C.keys, eps_meta, delta_meta)
    
    print(f"Table A Meta: a={meta_A.a}, b={meta_A.b}")
    print(f"Table B Meta: a={meta_B.a}, b={meta_B.b}")
    print(f"Table C Meta: a={meta_C.a}, b={meta_C.b}")

    # ---------------------------------------------------
    # [Phase 1] 单表差分隐私处理 (使用轻仓预算 eps_stage)
    # ---------------------------------------------------
    print("\n[Phase 1] 单表差分隐私处理 (Partition & Padding)...")
    parts_A, bucks_A = process_single_table("Table_A", table_A, D, eps_stage, delta_stage, eps_stage, delta_stage)
    parts_B, bucks_B = process_single_table("Table_B", table_B, D, eps_stage, delta_stage, eps_stage, delta_stage)
    parts_C, bucks_C = process_single_table("Table_C", table_C, D, eps_stage, delta_stage, eps_stage, delta_stage)

    # ---------------------------------------------------
    # [Phase 2] 执行 1st Join (使用轻仓预算 eps_stage)
    # ---------------------------------------------------
    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    print(f"[Join AB Meta] 动态推导结果 -> 敏感度 b={meta_AB.b}, 新频次上限 a={meta_AB.a}")
    
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    
    t_join1_start = time.time()
    parts_AB, bucks_AB = joiner_AB.run_join(
        parts_A, bucks_A,
        parts_B, bucks_B,
        dummy_key=-999, merge_factor=2
    )
    t_join1_end = time.time()
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # ---------------------------------------------------
    # [Phase 3] 执行 2nd Join (使用轻仓预算 eps_stage)
    # ---------------------------------------------------
    print("\n[Phase 3] 执行 2nd Join: (Table_A ⨝ Table_B) ⨝ Table_C ...")
    meta_ABC = meta_AB.join(meta_C)
    print(f"[Join ABC Meta] 动态推导结果 -> 敏感度 b={meta_ABC.b}, 新频次上限 a={meta_ABC.a}")
    
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_join(
        parts_AB, bucks_AB,
        parts_C, bucks_C,
        dummy_key=-999, merge_factor=3
    )
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终 3-Way Join (A ⨝ B ⨝ C) 结果报告")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)
    
    return parts_ABC, bucks_ABC

def extract_real_data_from_buckets(buckets, dummy_key=-999):
    """
    从 Bucket 列表中提取真实的 (key, freq) 数据。
    用于模拟获取 Join 的中间明文结果，以便重新进行 Partition。
    """
    freq_map = {}
    for bucket in buckets:
        for k, f in bucket:
            if k != dummy_key:
                freq_map[k] = freq_map.get(k, 0) + f
    return sorted(freq_map.items(), key=lambda x: x[0])

class DummyTable:
    """一个简单的占位类，用于适配 print_summary 函数打印报告"""
    def __init__(self, keys, sensitivity):
        self.keys = keys
        self.sensitivity = sensitivity

def run_3way_join_benchmark_repartition(D, N_A, N_B, N_C, base_epsilon=1.0, base_delta=1e-6, extra_epsilon=1.0, extra_delta=1e-6):
    """
    模拟 3-Way Join (Method 1: 重新 Partition 与 Bucketing)
    使用两套独立的隐私预算。
    """
    print(f"\n{'='*70}")
    print(f"Starting 3-Way Join Benchmark (Method 1: RE-PARTITION)")
    print(f"Base Budget (for A, B, C & Join1): eps={base_epsilon}, delta={base_delta}")
    print(f"Extra Budget (for Re-Part AB, Re-Buck AB & Join2): eps={extra_epsilon}, delta={extra_delta}")
    print(f"{'='*70}")

    # --- 1. 基础预算分配 (50% 给 Meta, 50% 给其余 4 个基础阶段) ---
    ratio_meta = 0.90
    ratio_others = (1.0 - ratio_meta) / 4.0

    eps_meta = base_epsilon * ratio_meta
    delta_meta = base_delta * ratio_meta
    eps_base_stage = base_epsilon * ratio_others
    delta_base_stage = base_delta * ratio_others

    # --- 2. 额外预算分配 (给重新 Partition, 重新 Bucket, 以及最终的 Join2 平分) ---
    eps_extra_stage = extra_epsilon / 3.0
    delta_extra_stage = extra_delta / 3.0

    # 生成数据
    shared_centers = np.random.randint(1, D, size=50)
    table_A = generate_simulation_table("Table_A", D, N_A, fixed_centers=shared_centers)
    table_B = generate_simulation_table("Table_B", D, N_B, fixed_centers=shared_centers)
    table_C = generate_simulation_table("Table_C", D, N_C, fixed_centers=shared_centers)
    
    # [Phase 0] Meta 计算
    print("\n[Phase 0] 计算基础表元数据...")
    meta_A = JoinMetadata.from_base_table(table_A.keys, eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table(table_B.keys, eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table(table_C.keys, eps_meta, delta_meta)

    # [Phase 1] 单表 Partition & Padding
    print("\n[Phase 1] 基础表 Partition & Padding...")
    parts_A, bucks_A = process_single_table("Table_A", table_A, D, eps_base_stage, delta_base_stage, eps_base_stage, delta_base_stage)
    parts_B, bucks_B = process_single_table("Table_B", table_B, D, eps_base_stage, delta_base_stage, eps_base_stage, delta_base_stage)
    parts_C, bucks_C = process_single_table("Table_C", table_C, D, eps_base_stage, delta_base_stage, eps_base_stage, delta_base_stage)

    # [Phase 2] 1st Join (A ⨝ B)
    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_base_stage, delta=delta_base_stage, sensitivity=meta_AB.b)
    t_join1_start = time.time()
    parts_AB, bucks_AB = joiner_AB.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
    t_join1_end = time.time()
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # ==========================================================
    # [Phase 2.5] 核心新增：提取中间结果并重新 Partition & Bucket
    # ==========================================================
    print("\n" + "*"*60)
    print("[Phase 2.5] 提取中间结果，消耗【额外预算】重新 Partition & Bucket")
    print("*"*60)
    
    # 1. 提取真实频次
    sorted_stats_AB = extract_real_data_from_buckets(bucks_AB)
    dummy_table_AB = DummyTable(keys=[k for k, f in sorted_stats_AB], sensitivity=meta_AB.b)
    
    # 2. 重新 Partition (注意：此处的 Sensitivity 是 Join 之后的 b)
    t_repart_start = time.time()
    part_algo_AB = PrivatePartition(
        epsilon=eps_extra_stage, 
        delta=delta_extra_stage, 
        domain_size=D, 
        sensitivity=meta_AB.b  # 极其关键：敏感度已膨胀！
    )
    new_parts_AB = part_algo_AB.run_partition(sorted_stats_AB)
    
    # 3. 重新 Bucket
    buck_algo_AB = BucketProcessor(
        partitions=new_parts_AB, 
        epsilon=eps_extra_stage, 
        delta=delta_extra_stage, 
        sensitivity=meta_AB.b
    )
    new_bucks_AB = buck_algo_AB.distribute_and_pad(sorted_stats_AB, dummy_key=-999)
    t_repart_end = time.time()
    
    print_summary("Re-Partitioned AB", new_parts_AB, new_bucks_AB, t_repart_end - t_repart_start, dummy_table_AB, D)

    # ==========================================================
    # [Phase 3] 2nd Join ((A ⨝ B)_new ⨝ C)
    # ==========================================================
    print("\n[Phase 3] 执行 2nd Join: (Table_A ⨝ Table_B)_repart ⨝ Table_C ...")
    meta_ABC = meta_AB.join(meta_C)
    
    joiner_ABC = DPJoiner(epsilon=eps_extra_stage, delta=delta_extra_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    # 注意这里输入的是 new_parts_AB 和 new_bucks_AB
    parts_ABC, bucks_ABC = joiner_ABC.run_join(
        new_parts_AB, new_bucks_AB,
        parts_C, bucks_C,
        merge_factor=3
    )
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终 3-Way Join (Method 1 重新切分) 结果报告")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)
    
    return parts_ABC, bucks_ABC


# 在 if __name__ == "__main__": 下方添加调用
if __name__ == "__main__":
    # 配置规模
    DOMAIN_SIZE = 100_0000   # D
    NUM_RECORDS_A = 1_000_0
    NUM_RECORDS_B = 800_0    
    NUM_RECORDS_C = 600_0  
    
    # run_3way_join_benchmark_repartition(
    #     D=DOMAIN_SIZE, 
    #     N_A=NUM_RECORDS_A, 
    #     N_B=NUM_RECORDS_B,
    #     N_C=NUM_RECORDS_C,
    #     base_epsilon=1.0,
    #     base_delta=1e-6,
    #     extra_epsilon=1.0,
    #     extra_delta=1e-6
    # )

    # 运行 3-way Join
    run_3way_join_benchmark(
        D=DOMAIN_SIZE, 
        N_A=NUM_RECORDS_A, 
        N_B=NUM_RECORDS_B,
        N_C=NUM_RECORDS_C,
        epsilon=1.0,
        delta=1e-6
    )

    # 隐私预算
    # EPSILON = 1.0
    # DELTA = 1e-6 
    
    # 运行双表 Join 模拟
    # j_parts, j_bucks = run_join_benchmark(
    #     D=DOMAIN_SIZE, 
    #     N_A=NUM_RECORDS_A, 
    #     N_B=NUM_RECORDS_B,
    #     epsilon=EPSILON,
    #     delta=DELTA
    # )  
    
     