import numpy as np
import time
import math
from collections import Counter

# 引入另外两个模块
from private_partition_M_truncated import PrivatePartition
# from private_partition import PrivatePartition
from bucket_mechanism import BucketProcessor

def generate_large_sparse_data(D, num_records, num_clusters=5):
    """
    生成大规模稀疏数据 (模拟真实分布)
    :param D: 全域大小 (1,000,000)
    :param num_records: 记录总数 (10,000)
    :param num_clusters: 数据聚集的热点数量
    """
    print(f"正在生成数据... (D={D}, Records={num_records})")
    data = []
    
    # 1. 生成热点数据 (80% 的数据集中在几个簇周围)
    records_in_clusters = int(num_records * 0.8)
    for _ in range(num_clusters):
        # 随机选一个中心点
        center = np.random.randint(1, D)
        # 生成标准差为 D*0.01 的高斯分布数据
        cluster_data = np.random.normal(loc=center, scale=D*0.005, size=records_in_clusters // num_clusters)
        # 取整并截断在 [1, D] 范围内
        cluster_data = np.clip(cluster_data, 1, D).astype(int)
        data.extend(cluster_data)
        
    # 2. 生成均匀噪声数据 (20% 的数据散落在各地)
    records_noise = num_records - len(data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    data.extend(uniform_data)
    
    # 排序方便查看 (实际 preprocess 会再排一次)
    data.sort()
    return data

def print_summary(partitions, final_buckets, elapsed_time, raw_N):
    """
    打印大规模测试的统计摘要
    """
    num_buckets = len(final_buckets)
    total_real_keys = 0
    total_dummy_items = 0
    empty_buckets = 0 # 指没有真实数据的桶 (纯噪声桶)
    
    for bucket in final_buckets:
        # bucket 里的结构是 (key, freq)
        real_in_bucket = sum(1 for k, f in bucket if k != -999)
        dummies_in_bucket = sum(1 for k, f in bucket if k == -999)
        
        total_real_keys += real_in_bucket
        total_dummy_items += dummies_in_bucket
        
        if real_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_keys + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    
    print(f"\n" + "="*40)
    print(f"      大规模测试统计报告      ")
    print(f"="*40)
    print(f"耗时: {elapsed_time:.4f} 秒")
    print(f"-"*40)
    print(f"原始 Key 数量 (Unique): {raw_N}")
    print(f"生成的 Bucket 总数:     {num_buckets}")
    print(f"平均 Bucket 大小:       {avg_bucket_size:.2f}")
    print(f"纯噪声桶 (Empty Buckets): {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*40)
    print(f"总 Item 数量:           {total_real_keys + total_dummy_items}")
    print(f"  - 真实数据:           {total_real_keys} (占比 {total_real_keys/(total_real_keys+total_dummy_items)*100:.2f}%)")
    print(f"  - Dummy 填充:         {total_dummy_items} (占比 {total_dummy_items/(total_real_keys+total_dummy_items)*100:.2f}%)")
    print(f"="*40 + "\n")

def run_simulation():
    # ==========================================
    # 1. 全局配置 (High Scale)
    # ==========================================
    D = 1_000_000         # 一百万全域
    N_approx = 10_000     # 约一万条数据
    
    total_epsilon = 1.0
    delta = 1e-7          # D 很大，Delta 需要更小以保证 log(D/delta) 有效
    
    eps_partition = total_epsilon * 0.5
    eps_bucket = total_epsilon * 0.5
    delta_partition = delta / 2
    delta_bucket = delta / 2

    # ==========================================
    # 2. 数据准备
    # ==========================================
    raw_data = generate_large_sparse_data(D, N_approx)
    
    # 实例化算法
    partition_algo = PrivatePartition(eps_partition, delta_partition, D)
    
    start_time = time.time()
    
    # 预处理
    sorted_stats = partition_algo.preprocess_data(raw_data)
    real_unique_keys = len(sorted_stats)
    
    print(f"数据预处理完成。Unique Keys: {real_unique_keys}. 开始算法...")

    # ==========================================
    # 3. 运行 Partition
    # ==========================================
    partitions = partition_algo.run_partition(sorted_stats)
    print(f">> Partition 完成。切分出 {len(partitions)} 个区间。")
    
    # 检查一下前几个和后几个区间，看看是否合理
    print(f"   Head Partitions: {partitions[:3]}")
    print(f"   Tail Partitions: {partitions[-3:]}")

    # ==========================================
    # 4. 运行 Bucket Mechanism
    # ==========================================
    bucket_processor = BucketProcessor(partitions, eps_bucket, delta_bucket)
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    
    end_time = time.time()

    # ==========================================
    # 5. 结果展示
    # ==========================================
    print_summary(partitions, final_buckets, end_time - start_time, real_unique_keys)
    
    # 抽查一个中间的 Bucket 看看细节
    if len(final_buckets) > 10:
        sample_idx = len(final_buckets) // 2
        b = final_buckets[sample_idx]
        real_cnt = sum(1 for k,f in b if k!=-999)
        dummy_cnt = len(b) - real_cnt
        print(f"[抽查 Bucket #{sample_idx}] Range: {partitions[sample_idx]}")
        print(f"   Real: {real_cnt}, Dummy: {dummy_cnt}, Total: {len(b)}")

if __name__ == "__main__":
    run_simulation()


'''
(.venv) parallels@ubuntu-linux-2404:~/Join$ /home/parallels/Join/.venv/bin/python /home/parallels/Join/main_simulation_D_1m.py
正在生成数据... (D=1000000, Records=10000)
[Init] Params: T=119.73, NoiseBound(M)=33.62
数据预处理完成。Unique Keys: 9616. 开始算法...
>> Partition 完成。切分出 90 个区间。
   Head Partitions: [(1, np.int64(59930)), (np.int64(59931), np.int64(116186)), (np.int64(116187), np.int64(171767))]
   Tail Partitions: [(np.int64(909687), np.int64(919224)), (np.int64(919225), np.int64(972513)), (np.int64(972514), 1000000)]
开始处理 90 个 Bucket (Adding Noise & Dummy)...
  Bucket 0 (Range (1, np.int64(59930))): Real=105, Noise=39, Total=144
  Bucket 1 (Range (np.int64(59931), np.int64(116186))): Real=113, Noise=36, Total=149
  Bucket 2 (Range (np.int64(116187), np.int64(171767))): Real=113, Noise=36, Total=149

========================================
      大规模测试统计报告      
========================================
耗时: 0.0345 秒
----------------------------------------
原始 Key 数量 (Unique): 9616
生成的 Bucket 总数:     90
平均 Bucket 大小:       143.07
纯噪声桶 (Empty Buckets): 0 (占比 0.00%)
----------------------------------------
总 Item 数量:           12876
  - 真实数据:           9616 (占比 74.68%)
  - Dummy 填充:         3260 (占比 25.32%)
========================================

[抽查 Bucket #45] Range: (np.int64(667146), np.int64(668011))
   Real: 109, Dummy: 35, Total: 144
'''