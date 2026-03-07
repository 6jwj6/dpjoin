import numpy as np
import time
from data_structure import Table
from private_partition import PrivatePartition
from bucket_mechanism import BucketProcessor

def generate_simulation_table(name: str, D: int, num_records: int, sensitivity: int = 1, num_clusters: int = 50) -> Table:
    """
    [高性能数据生成器]
    生成符合大规模稀疏分布的 Table 对象，支持自定义敏感度。
    """
    print(f"[{name}] Generating Data... D={D:,}, N={num_records:,}, mu={sensitivity}")
    t0 = time.time()
    
    # 1. 生成热点数据
    records_in_clusters = int(num_records * 0.8)
    centers = np.random.randint(1, D, size=num_clusters)
    cluster_size = records_in_clusters // num_clusters
    
    cluster_data = np.random.normal(loc=centers[:, None], scale=D*0.0001, size=(num_clusters, cluster_size))
    cluster_data = cluster_data.flatten()
    
    # 2. 生成均匀分布数据
    records_noise = num_records - len(cluster_data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    
    # 3. 合并
    all_keys = np.concatenate([cluster_data, uniform_data])
    all_keys = np.clip(all_keys, 1, D).astype(int)
    
    # 4. 创建 Table 对象
    payloads = np.full(len(all_keys), None, dtype=object)
    table = Table(name=name, keys=all_keys, payloads=payloads, sensitivity=sensitivity)
    
    print(f"[{name}] Done. Time: {time.time() - t0:.4f}s. TrueMaxFreq: {table.true_max_freq}")
    return table

def print_summary(partitions, final_buckets, elapsed_time, table: Table, D: int):
    """
    打印详细统计报告
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
    sparsity = 1.0 - (total_real_keys / D)
    
    print(f"\n" + "="*50)
    print(f"      差分隐私 Partition & Bucket 压力测试报告      ")
    print(f"="*50)
    print(f"全域大小 (D):           {D:,}")
    print(f"敏感度 (mu):            {table.sensitivity}")
    print(f"非零 Key (Unique):      {total_real_keys:,}")
    print(f"稀疏度 (Sparsity):      {sparsity:.6%}")
    print(f"-"*50)
    print(f"算法总耗时:             {elapsed_time:.4f} 秒")
    print(f"-"*50)
    print(f"生成的 Bucket 总数:     {num_buckets:,}")
    print(f"平均项数/Bucket:        {avg_bucket_size:.2f} items")
    print(f"纯噪声桶 (Empty):       {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*50)
    print(f"数据总量 (Real+Dummy):  {total_real_keys + total_dummy_items:,}")
    print(f"  - 真实数据 (Real):    {total_real_keys:,}")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,} (Ratio: {total_dummy_items/total_real_keys:.2f}x)")
    print(f"="*50 + "\n")

def run_benchmark(D, N, sensitivity=1, epsilon=1.0, delta=1e-9):
    """
    核心运行逻辑，修正了 delta 的预算分配以及 BucketProcessor 的实例化顺序
    """
    # --- 1. 隐私预算分配 ---
    eps_meta = epsilon * 0.1
    eps_partition = epsilon * 0.45
    eps_bucket = epsilon * 0.45
    
    delta_meta = delta * 0.1
    delta_partition = delta * 0.45
    delta_bucket = delta * 0.45

    # --- 2. 数据准备与元数据加噪 ---
    table = generate_simulation_table("BenchTable", D, N, sensitivity=sensitivity)
    
    print(f">> Privatizing Metadata (eps={eps_meta:.2f}, delta={delta_meta:.2e})...")
    table.privatize_metadata(eps_meta, delta_meta)
    
    # --- 3. 执行 Partition 流程 ---
    partition_algo = PrivatePartition(
        eps_partition, 
        delta_partition, 
        D, 
        sensitivity=table.sensitivity
    )
    
    print("Pre-processing (Counting & Sorting)...")
    t_start = time.time()
    sorted_stats = partition_algo.preprocess_data(table.keys)
    
    print(f"Running Partition (mu={table.sensitivity}, eps={eps_partition:.2f})...")
    partitions = partition_algo.run_partition(sorted_stats)
    print(f">> Partition finished. Created {len(partitions):,} intervals.")

    # --- 4. 实例化 BucketProcessor (关键修正：在拿到 partitions 之后实例化) ---
    # NONONONONONO NEED，这样内部的 delta_local 才能根据真实的 num_buckets 进行分摊
    print(f"Initializing Bucket Mechanism with {len(partitions)} buckets...")
    bucket_processor = BucketProcessor(
        partitions, 
        eps_bucket, 
        delta_bucket, 
        sensitivity=table.sensitivity
    )
    
    # --- 5. 执行 Padding 流程 ---
    print(f"Running Bucket Mechanism (eps={eps_bucket:.2f})...")
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
    SENSITIVITY = 1             # 模拟敏感度
    
    # 运行
    buckets, parts = run_benchmark(D=DOMAIN_SIZE, N=NUM_RECORDS, sensitivity=SENSITIVITY)