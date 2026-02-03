import numpy as np
import time
from data_structure import Table
from private_partition import PrivatePartition
from bucket_mechanism import BucketProcessor

def generate_simulation_table(name: str, D: int, num_records: int, num_clusters: int = 50) -> Table:
    """
    [高性能数据生成器]
    生成符合大规模稀疏分布的 Table 对象。
    """
    print(f"[{name}] Generating Data (Vectorized)... D={D:,}, N={num_records:,}")
    t0 = time.time()
    
    # 1. 生成热点数据 (80% 的数据集中在几个簇)
    records_in_clusters = int(num_records * 0.8)
    centers = np.random.randint(1, D, size=num_clusters)
    cluster_size = records_in_clusters // num_clusters
    
    # 使用 numpy 广播生成高斯分布数据
    # scale=D*0.0001 意味着数据非常集中
    cluster_data = np.random.normal(loc=centers[:, None], scale=D*0.0001, size=(num_clusters, cluster_size))
    cluster_data = cluster_data.flatten()
    
    # 2. 生成均匀噪声数据 (20% 的数据散落在各地)
    records_noise = num_records - len(cluster_data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    
    # 3. 合并并处理边界
    all_keys = np.concatenate([cluster_data, uniform_data])
    all_keys = np.clip(all_keys, 1, D).astype(int)
    
    # 4. 创建 Table 对象
    # Payloads 暂时为空
    payloads = np.full(len(all_keys), None, dtype=object)
    table = Table(name=name, keys=all_keys, payloads=payloads)
    
    print(f"[{name}] Generation Done. Time: {time.time() - t0:.4f}s. TrueMaxFreq: {table.true_max_freq}")
    return table

def print_summary(partitions, final_buckets, elapsed_time, table: Table, D: int):
    """
    打印详细的统计报告
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
    raw_N = len(table.keys)
    unique_keys = total_real_keys # 也就是 preprocess 之后的 key 数量
    sparsity = 1.0 - (unique_keys / D)
    
    print(f"\n" + "="*50)
    print(f"      差分隐私 Partition & Bucket 压力测试报告      ")
    print(f"="*50)
    print(f"全域大小 (D):           {D:,}")
    print(f"总记录数 (Rows):        {raw_N:,}")
    print(f"非零 Key (Unique):      {unique_keys:,}")
    print(f"稀疏度 (Sparsity):      {sparsity:.6%}")
    print(f"-"*50)
    print(f"算法总耗时:             {elapsed_time:.4f} 秒")
    print(f"-"*50)
    print(f"生成的 Bucket 总数:     {num_buckets:,}")
    print(f"平均 Bucket 大小:       {avg_bucket_size:.2f} items")
    print(f"纯噪声桶 (Empty):       {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*50)
    print(f"数据总量 (Real+Dummy):  {total_real_keys + total_dummy_items:,}")
    print(f"  - 真实数据 (Real):    {total_real_keys:,}")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,} (Ratio: {total_dummy_items/total_real_keys:.2f}x)")
    print(f"="*50 + "\n")

def run_benchmark(D, N, epsilon=1.0, delta=1e-9):
    """
    核心运行逻辑，接受参数输入
    """
    # 参数分配
    eps_partition = epsilon * 0.5
    eps_bucket = epsilon * 0.5
    delta_partition = delta / 2
    delta_bucket = delta / 2

    # 1. 生成数据 (Table)
    table = generate_simulation_table("BenchTable", D, N)
    
    # 2. 实例化算法
    partition_algo = PrivatePartition(eps_partition, delta_partition, D)
    bucket_processor = BucketProcessor(None, eps_bucket, delta_bucket) # partitions稍后填入
    
    # 3. 预处理
    print("Pre-processing (Counting & Sorting)...")
    t_start = time.time()
    sorted_stats = partition_algo.preprocess_data(table.keys)
    
    # 4. 运行 Partition
    print("Running Partition (Gap-Truncated SVT)...")
    partitions = partition_algo.run_partition(sorted_stats)
    print(f">> Partition finished. Created {len(partitions):,} intervals.")

    # 5. 运行 Bucket Mechanism
    print("Running Bucket Mechanism (Padding)...")
    # 更新 processor 的 partitions
    bucket_processor.partitions = partitions 
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    
    t_end = time.time()
    
    # 6. 打印报告
    print_summary(partitions, final_buckets, t_end - t_start, table, D)
    
    return final_buckets, partitions

# ==========================================
#  在此处修改参数直接运行
# ==========================================
if __name__ == "__main__":
    # 配置你的测试规模
    DOMAIN_SIZE = 100_000_000   # D: 一亿
    NUM_RECORDS = 1_000_000     # N: 一百万
    
    # 运行
    buckets, parts = run_benchmark(D=DOMAIN_SIZE, N=NUM_RECORDS)
    
    # 简单的抽查逻辑
    if len(buckets) > 5:
        idx = 0 # 查看第一个桶
        print(f"Sample Bucket #{idx} Range: {parts[idx]}")
        print(f"Size: {len(buckets[idx])} (Real + Dummy)")