import numpy as np
from collections import Counter

# 引入另外两个模块
# 确保 private_partition.py 和 bucket_mechanism.py 在同一目录下
from private_partition_M_truncated import PrivatePartition
# from private_partition import PrivatePartition
from bucket_mechanism import BucketProcessor

def generate_mock_data(domain_size=100):
    """
    生成模拟数据：包含密集区、稀疏区和大 Gap
    """
    data = []
    # 区域 1: [1-10] 密集区
    for k in range(1, 11):
        data.extend([k] * np.random.randint(5, 15))
    
    # 区域 2: [40-45] 稀疏区 (中间有个大 Gap 11-39)
    for k in range(40, 46):
        data.extend([k] * np.random.randint(1, 5))
        
    # 区域 3: [90-100] 超级密集区 (另一个 Gap 46-89)
    for k in range(90, 101):
        data.extend([k] * np.random.randint(20, 50))
        
    return data

def run_simulation():
    # ==========================================
    # 1. 全局配置
    # ==========================================
    D = 100               # 全域大小
    total_epsilon = 1.0   # 总隐私预算
    delta = 1e-5          # 失败概率
    
    # 预算分配 (简单起见，对半均分)
    # Partition 消耗一半，Bucket Padding 消耗一半
    eps_partition = total_epsilon * 0.5
    eps_bucket = total_epsilon * 0.5
    
    # Bucket Padding 还需要 delta，通常沿用全局 delta 或根据组合定理分配
    delta_partition = delta / 2
    delta_bucket = delta / 2

    print(f"=== 模拟开始 ===")
    print(f"参数: D={D}, Total Epsilon={total_epsilon}, Delta={delta}\n")

    # ==========================================
    # 2. 数据准备
    # ==========================================
    raw_data = generate_mock_data(D)
    print(f"生成原始数据: {len(raw_data)} 条记录")
    
    # 实例化 Partition 算法
    partition_algo = PrivatePartition(eps_partition, delta_partition, D)
    
    # 数据预处理 (统计 key, freq)
    sorted_stats = partition_algo.preprocess_data(raw_data)
    # sorted_stats 结构: [(key, freq), ...]

    # ==========================================
    # 3. 运行 Partition (Gap-Truncated SVT)
    # ==========================================
    print(f"\n>> Step 1: 运行 Partition 算法...")
    # 获取切分出的区间列表，例如 [(1, 10), (40, 60), ...]
    partitions = partition_algo.run_partition(sorted_stats)
    
    print(f"Partition 完成，共划分为 {len(partitions)} 个区间:")
    for p in partitions:
        print(f"  区间: {p}")

    # ==========================================
    # 4. 运行 Bucket Mechanism (Padding & Dummy)
    # ==========================================
    print(f"\n>> Step 2: 运行 Bucket Padding (Shifted Truncated Geo)...")
    
    # 实例化 Bucket 处理器
    bucket_processor = BucketProcessor(partitions, eps_bucket, delta_bucket)
    
    # 分发数据并填充 Dummy
    # 这里我们用 dummy_key = -999 来标记 dummy 数据
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)

    # ==========================================
    # 5. 结果展示与对比
    # ==========================================
    print(f"\n=== 最终结果展示 ===")
    
    for i, bucket in enumerate(final_buckets):
        start, end = partitions[i]
        
        # 统计 Bucket 内部情况
        real_count = sum(1 for k, f in bucket if k != -999) # 统计有多少个真实 Key
        dummy_count = sum(1 for k, f in bucket if k == -999) # 统计 Dummy 数量
        
        # 计算真实负载 (Sum of Freq) vs 填充后的感知负载
        # 注意：bucket 里的结构是 (key, freq)
        # Dummy 的 freq 通常为 0，所以 sum(freq) 不变，但 items 数量变了
        real_load_sum = sum(f for k, f in bucket if k != -999)
        
        print(f"Bucket {i} [Range {start}-{end}]:")
        print(f"  - 真实 Key 数量: {real_count}")
        print(f"  - 填充 Dummy 数量: {dummy_count} (Noise)")
        print(f"  - 总 Item 数量: {len(bucket)}")
        print(f"  - 数据预览 (前5个): {bucket[:5]} ...")
        
        # 验证安全性：
        # 如果这是一个 Gap 区间（比如 partition 切出来的区间里本来就没有数据），
        # 我们应该能看到 real_count=0，但 dummy_count > 0 (由 k0 决定)
        if real_count == 0:
            print(f"  [观察]: 这是一个被噪声掩盖的空桶/稀疏桶！")

if __name__ == "__main__":
    run_simulation()


'''
(.venv) parallels@ubuntu-linux-2404:~/Join$ /home/parallels/Join/.venv/bin/python /home/parallels/Join/main_simulation_D_100.py
=== 模拟开始 ===
参数: D=100, Total Epsilon=1.0, Delta=1e-05

生成原始数据: 482 条记录
[Init] Params: T=64.47, NoiseBound(M)=24.41

>> Step 1: 运行 Partition 算法...
Partition 完成，共划分为 7 个区间:
  区间: (1, 8)
  区间: (9, 90)
  区间: (91, 92)
  区间: (93, 94)
  区间: (95, 96)
  区间: (97, 99)
  区间: (100, 100)

>> Step 2: 运行 Bucket Padding (Shifted Truncated Geo)...
开始处理 7 个 Bucket (Adding Noise & Dummy)...
  Bucket 0 (Range (1, 8)): Real=8, Noise=24, Total=32
  Bucket 1 (Range (9, 90)): Real=9, Noise=27, Total=36
  Bucket 2 (Range (91, 92)): Real=2, Noise=25, Total=27

=== 最终结果展示 ===
Bucket 0 [Range 1-8]:
  - 真实 Key 数量: 8
  - 填充 Dummy 数量: 24 (Noise)
  - 总 Item 数量: 32
  - 数据预览 (前5个): [(1, 9), (2, 13), (3, 9), (4, 6), (5, 6)] ...
Bucket 1 [Range 9-90]:
  - 真实 Key 数量: 9
  - 填充 Dummy 数量: 27 (Noise)
  - 总 Item 数量: 36
  - 数据预览 (前5个): [(9, 8), (10, 11), (40, 3), (41, 2), (42, 1)] ...
Bucket 2 [Range 91-92]:
  - 真实 Key 数量: 2
  - 填充 Dummy 数量: 25 (Noise)
  - 总 Item 数量: 27
  - 数据预览 (前5个): [(91, 20), (92, 46), (-999, 0), (-999, 0), (-999, 0)] ...
Bucket 3 [Range 93-94]:
  - 真实 Key 数量: 2
  - 填充 Dummy 数量: 27 (Noise)
  - 总 Item 数量: 29
  - 数据预览 (前5个): [(93, 30), (94, 41), (-999, 0), (-999, 0), (-999, 0)] ...
Bucket 4 [Range 95-96]:
  - 真实 Key 数量: 2
  - 填充 Dummy 数量: 27 (Noise)
  - 总 Item 数量: 29
  - 数据预览 (前5个): [(95, 21), (96, 45), (-999, 0), (-999, 0), (-999, 0)] ...
Bucket 5 [Range 97-99]:
  - 真实 Key 数量: 3
  - 填充 Dummy 数量: 27 (Noise)
  - 总 Item 数量: 30
  - 数据预览 (前5个): [(97, 29), (98, 27), (99, 45), (-999, 0), (-999, 0)] ...
Bucket 6 [Range 100-100]:
  - 真实 Key 数量: 1
  - 填充 Dummy 数量: 25 (Noise)
  - 总 Item 数量: 26
  - 数据预览 (前5个): [(100, 37), (-999, 0), (-999, 0), (-999, 0), (-999, 0)] ...
'''