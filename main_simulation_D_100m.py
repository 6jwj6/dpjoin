import numpy as np
import time
import math
from collections import Counter

# 引入另外两个模块
from private_partition_M_truncated import PrivatePartition
# from private_partition import PrivatePartition
from bucket_mechanism import BucketProcessor

def generate_huge_data_vectorized(D, num_records, num_clusters=20):
    """
    [高性能版] 生成大规模稀疏数据
    针对 100万+ 记录进行了 Numpy 向量化优化，生成速度极快。
    """
    print(f"正在生成数据 (Vectorized)... D={D:,}, Records={num_records:,}")
    t0 = time.time()
    
    # 1. 生成热点数据 (80% 的数据)
    # 使用 concatenate 一次性生成，避免循环 extend
    records_in_clusters = int(num_records * 0.8)
    # 随机选择簇中心
    centers = np.random.randint(1, D, size=num_clusters)
    # 每个簇的大小
    cluster_size = records_in_clusters // num_clusters
    
    # 生成所有簇的数据 (高斯分布)
    # scale 设置为 D 的万分之一，模拟比较集中的热点
    cluster_data = np.random.normal(loc=centers[:, None], scale=D*0.0001, size=(num_clusters, cluster_size))
    cluster_data = cluster_data.flatten()
    
    # 2. 生成均匀噪声数据 (20% 的数据)
    records_noise = num_records - len(cluster_data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    
    # 3. 合并并处理边界
    all_data = np.concatenate([cluster_data, uniform_data])
    # 截断到 [1, D] 并转为整数
    all_data = np.clip(all_data, 1, D).astype(int)
    
    # 转为 Python list (如果后续需要) 或者直接给 Counter (Counter 接受 numpy array)
    print(f"数据生成耗时: {time.time() - t0:.4f} 秒")
    return all_data

def print_summary(partitions, final_buckets, elapsed_time, raw_N, D):
    """
    亿级规模测试的统计报告
    """
    num_buckets = len(final_buckets)
    total_real_keys = 0
    total_dummy_items = 0
    empty_buckets = 0 
    
    for bucket in final_buckets:
        # bucket: [(key, freq), ...]
        real_in_bucket = sum(1 for k, f in bucket if k != -999)
        dummies_in_bucket = len(bucket) - real_in_bucket
        
        total_real_keys += real_in_bucket
        total_dummy_items += dummies_in_bucket
        
        if real_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_keys + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    sparsity = 1.0 - (raw_N / D)
    
    print(f"\n" + "="*50)
    print(f"      1亿全域 x 100万数据 压力测试报告      ")
    print(f"="*50)
    print(f"全域大小 (D):           {D:,}")
    print(f"非零 Key 数量 (N):      {raw_N:,}")
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

def run_simulation():
    # ==========================================
    # 1. 亿级参数配置
    # ==========================================
    D = 100_000_000       # 一亿
    N_approx = 1_000_000  # 一百万
    
    # 簇的数量增加一点，模拟更真实的数据分布
    num_clusters = 50     
    
    total_epsilon = 1.0
    delta = 1e-9          # D 很大，Delta 设得更小一点比较安全
    
    eps_partition = total_epsilon * 0.5
    eps_bucket = total_epsilon * 0.5
    delta_partition = delta / 2
    delta_bucket = delta / 2

    # ==========================================
    # 2. 数据准备
    # ==========================================
    # 使用优化后的生成器
    raw_data = generate_huge_data_vectorized(D, N_approx, num_clusters)
    
    partition_algo = PrivatePartition(eps_partition, delta_partition, D)
    
    print("开始预处理 (Counter & Sorting)...")
    t_prep_start = time.time()
    # Counter 处理 100万数据非常快
    sorted_stats = partition_algo.preprocess_data(raw_data)
    real_unique_keys = len(sorted_stats)
    print(f"预处理完成。耗时: {time.time() - t_prep_start:.4f} 秒。Unique Keys: {real_unique_keys:,}")

    # ==========================================
    # 3. 运行算法核心 (计时开始)
    # ==========================================
    print(f"\n>> 启动 Partition (Gap-Truncated SVT)...")
    start_time = time.time()
    
    # 这里是核心：虽然 D 是 1亿，但循环次数只取决于 real_unique_keys (约100万)
    partitions = partition_algo.run_partition(sorted_stats)
    
    print(f">> Partition 完成。切分出 {len(partitions):,} 个区间。")
    
    print(f">> 启动 Bucket Mechanism (Padding)...")
    bucket_processor = BucketProcessor(partitions, eps_bucket, delta_bucket)
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    
    end_time = time.time()
    
    # ==========================================
    # 4. 结果展示
    # ==========================================
    print_summary(partitions, final_buckets, end_time - start_time, real_unique_keys, D)
    
    # 抽查几个桶
    print("抽查 Bucket 样本:")
    if len(final_buckets) > 5:
        # 取中间的一个桶
        idx = len(final_buckets) // 2
        b = final_buckets[idx]
        real_cnt = sum(1 for k,f in b if k!=-999)
        print(f"  [Bucket #{idx}] Range: {partitions[idx]}")
        print(f"    Real Keys: {real_cnt}, Total Items: {len(b)}")
        # 看看第一个桶（通常很密）
        print(f"  [Bucket #0] Range: {partitions[0]}")
        print(f"    Total Items: {len(final_buckets[0])}")

if __name__ == "__main__":
    run_simulation()



'''
(.venv) parallels@ubuntu-linux-2404:~/Join$ /home/parallels/Join/.venv/bin/python /home/parallels/Join/main_simulation_D_100m.py
正在生成数据 (Vectorized)... D=100,000,000, Records=1,000,000
数据生成耗时: 0.0201 秒
[Init] Params: T=156.58, NoiseBound(M)=42.83
开始预处理 (Counter & Sorting)...
预处理完成。耗时: 0.2649 秒。Unique Keys: 845,625

>> 启动 Partition (Gap-Truncated SVT)...
>> Partition 完成。切分出 6,600 个区间。
>> 启动 Bucket Mechanism (Padding)...
开始处理 6600 个 Bucket (Adding Noise & Dummy)...
  Bucket 0 (Range (1, np.int64(70167))): Real=142, Noise=38, Total=180
  Bucket 1 (Range (np.int64(70168), np.int64(152910))): Real=143, Noise=44, Total=187
  Bucket 2 (Range (np.int64(152911), np.int64(224182))): Real=142, Noise=44, Total=186

==================================================
      1亿全域 x 100万数据 压力测试报告      
==================================================
全域大小 (D):           100,000,000
非零 Key 数量 (N):      845,625
稀疏度 (Sparsity):      99.154375%
--------------------------------------------------
算法总耗时:             2.1992 秒
--------------------------------------------------
生成的 Bucket 总数:     6,600
平均 Bucket 大小:       173.12 items
纯噪声桶 (Empty):       0 (占比 0.00%)
--------------------------------------------------
数据总量 (Real+Dummy):  1,142,612
  - 真实数据 (Real):    845,625
  - 噪音填充 (Dummy):   296,987 (Ratio: 0.35x)
==================================================

抽查 Bucket 样本:
  [Bucket #3300] Range: (np.int64(47292660), np.int64(47292947))
    Real Keys: 120, Total Items: 165
  [Bucket #0] Range: (1, np.int64(70167))
    Total Items: 180
'''