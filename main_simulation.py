import numpy as np
import time
# 假设 data_structure.py 中有 Table 类
from data_structure import Table 
# 假设 private_partition.py 中是我们最新的 PrivatePartitionOffline 类
from private_partition import PrivatePartitionOffline as PrivatePartition 
# 假设 bucket_mechanism.py 中是我们最新的 BucketProcessor 类
from bucket_mechanism import BucketProcessor

def generate_simulation_table(name: str, D: int, num_records: int, sensitivity: int = 1, num_clusters: int = 50) -> Table:
    """
    [高性能数据生成器]
    生成符合大规模稀疏分布的 Table 对象，支持自定义敏感度。
    """
    print(f"[{name}] Generating Data... D={D:,}, N={num_records:,}, sens={sensitivity}")
    t0 = time.time()
    
    # 1. 生成热点数据 (80% 数据集中在少数 Cluster)
    records_in_clusters = int(num_records * 0.8)
    centers = np.random.randint(1, D, size=num_clusters)
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
    
    # 4. 创建 Table 对象
    # 为了模拟轻量化，Payload 设为简单的 int 1，或者 None
    # payloads = np.ones(len(all_keys), dtype=int) 
    payloads = [None] * len(all_keys) # 保持原样也可以，只要内存够
    
    table = Table(name=name, keys=all_keys, payloads=payloads, sensitivity=sensitivity)
    
    print(f"[{name}] Done. Time: {time.time() - t0:.4f}s. MaxKey: {np.max(all_keys)}")
    return table

def print_summary(partitions, final_buckets, elapsed_time, table: Table, D: int):
    """
    打印详细统计报告
    """
    num_buckets = len(final_buckets)
    total_real_keys = 0
    total_dummy_items = 0
    empty_buckets = 0
    
    # 统计每个 Bucket 的情况
    for bucket in final_buckets:
        # 假设 bucket 是 list of (key, freq) tuples
        real_in_bucket = sum(1 for k, f in bucket if k != -999)
        dummies_in_bucket = len(bucket) - real_in_bucket
        
        total_real_keys += real_in_bucket
        total_dummy_items += dummies_in_bucket
        
        if real_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_keys + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    # 稀疏度定义：全域中有多少比例的 Key 是没有数据的
    # 注意：这里的 total_real_keys 是记录数(N)，不是 Unique Key 数
    # 如果要算 Unique Key 覆盖率，需要在外面传进来。这里暂时用 Partition 数量估算或者忽略。
    
    print(f"\n" + "="*60)
    print(f"      差分隐私 Partition & Bucket 压力测试报告      ")
    print(f"="*60)
    print(f"参数设置:")
    print(f"  - Domain Size (D):      {D:,}")
    print(f"  - Sensitivity (mu):     {table.sensitivity}")
    print(f"  - Input Records (N):    {len(table.keys):,}")
    print(f"-"*60)
    print(f"性能指标:")
    print(f"  - 算法总耗时:           {elapsed_time:.4f} 秒")
    print(f"  - Throughput:           {len(table.keys)/elapsed_time:.0f} records/sec")
    print(f"-"*60)
    print(f"Partition 结果:")
    print(f"  - 生成区间数:           {len(partitions):,}")
    print(f"  - 纯噪声桶 (Empty):     {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*60)
    print(f"Bucket 填充结果:")
    print(f"  - 总项数 (Real+Dummy):  {total_real_keys + total_dummy_items:,}")
    print(f"  - 真实数据 (Real):      {total_real_keys:,}")
    print(f"  - 噪音填充 (Dummy):     {total_dummy_items:,}")
    print(f"  - 膨胀系数 (Ratio):     {(total_real_keys + total_dummy_items)/total_real_keys:.2f}x")
    print(f"  - 平均 Bucket 大小:     {avg_bucket_size:.2f} items")
    print(f"="*60 + "\n")

def run_benchmark(D, N, sensitivity=1, epsilon=1.0, delta=1e-6):
    """
    核心运行逻辑
    """
    print(f"Starting Benchmark: eps={epsilon}, delta={delta}")

    # --- 1. 隐私预算分配 ---
    # Partition 需要较大的 Delta 来保证 Offline 阈值 T 足够小(Utility)但又足够安全(Privacy)
    # Bucket 需要 Delta 来做截断
    eps_meta = epsilon * 0.1
    eps_partition = epsilon * 0.45
    eps_bucket = epsilon * 0.45
    
    delta_meta = delta * 0.1
    delta_partition = delta * 0.45
    delta_bucket = delta * 0.45

    # --- 2. 数据准备 ---
    table = generate_simulation_table("BenchTable", D, N, sensitivity=sensitivity)
    
    # 模拟 Meta Data 隐私化 (这里先跳过具体实现，假设消耗了预算)
    print(f">> Privatizing Metadata (eps={eps_meta:.2f}, delta={delta_meta:.2e})...")
    
    # --- 3. 执行 Partition 流程 ---
    # 使用最新的 PrivatePartitionOffline
    partition_algo = PrivatePartition(
        epsilon=eps_partition, 
        delta=delta_partition, # 这里的 delta 用于计算隐私阈值 T
        domain_size=D, 
        sensitivity=table.sensitivity
    )
    
    print("Pre-processing (Counting & Sorting)...")
    t_start = time.time()
    # 这里的 preprocess 应该是统计每个 Key 的频次
    sorted_stats = partition_algo.preprocess_data(table.keys)
    
    print(f"Running Partition (mu={table.sensitivity}, eps={eps_partition:.2f})...")
    partitions = partition_algo.run_partition(sorted_stats)
    print(f">> Partition finished. Created {len(partitions):,} intervals.")

    # --- 4. 实例化 BucketProcessor ---
    # 这里的 delta_bucket 直接传入，内部不需要再除以 bucket 数量 (Parallel Composition)
    print(f"Initializing Bucket Mechanism with {len(partitions)} buckets...")
    bucket_processor = BucketProcessor(
        partitions=partitions, 
        epsilon=eps_bucket, 
        delta=delta_bucket, 
        sensitivity=table.sensitivity
    )
    
    # --- 5. 执行 Padding 流程 ---
    print(f"Running Bucket Mechanism (eps={eps_bucket:.2f})...")
    # 注意：sorted_stats 是 (key, freq) 列表，BucketProcessor 会根据 partition 区间分发
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    
    t_end = time.time()
    
    # 6. 打印报告
    print_summary(partitions, final_buckets, t_end - t_start, table, D)
    
    return final_buckets, partitions

# ==========================================
# 修改此处参数运行
# ==========================================
if __name__ == "__main__":
    # 配置规模
    DOMAIN_SIZE = 100_000_000   # D: 一亿
    NUM_RECORDS = 1_000_000     # N: 一百万
    SENSITIVITY = 1             # 模拟敏感度 (Count)
    
    # 隐私预算
    EPSILON = 1.0
    DELTA = 1e-6 # 对于 1亿 Domain，1e-6 是合理的
    
    # 运行
    buckets, parts = run_benchmark(
        D=DOMAIN_SIZE, 
        N=NUM_RECORDS, 
        sensitivity=SENSITIVITY,
        epsilon=EPSILON,
        delta=DELTA
    )