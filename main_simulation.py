import numpy as np
import time
# 假设 data_structure.py 中有 Table 类
from data_structure import Table 
# 假设 private_partition.py 中是我们最新的 PrivatePartitionOffline 类
from private_partition import PrivatePartitionOffline as PrivatePartition 
# 假设 bucket_mechanism.py 中是我们最新的 BucketProcessor 类
from bucket_mechanism import BucketProcessor
# 引入最新编写的 Join 模块
from join_mechanism import DPJoiner

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

# ==========================================
# 修改此处参数运行
# ==========================================
if __name__ == "__main__":
    # 配置规模
    DOMAIN_SIZE = 100_000_000   # D: 一亿
    NUM_RECORDS_A = 1_000_000   # 表A: 一百万
    NUM_RECORDS_B = 800_000     # 表B: 八十万
    
    # 隐私预算
    EPSILON = 1.0
    DELTA = 1e-6 
    
    # 运行双表 Join 模拟
    j_parts, j_bucks = run_join_benchmark(
        D=DOMAIN_SIZE, 
        N_A=NUM_RECORDS_A, 
        N_B=NUM_RECORDS_B,
        epsilon=EPSILON,
        delta=DELTA
    )   