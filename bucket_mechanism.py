import numpy as np
import math

class ShiftedTruncatedGeometricMechanism:
    """
    实现了 Definition 3.6 中的 Shifted and Truncated Geometric Distribution。
    用于生成非负的、有界的离散噪声，通常用于确定 Padding 的大小。
    """
    def __init__(self, epsilon, delta, sensitivity=1):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # 1. 计算 k0 (根据 Definition 3.6)
        # k0 是最小的正整数，使得 Pr[|Geom| >= k0] <= delta
        # 公式: k0 approx (Delta / epsilon) * ln(2 / delta)
        self.k0 = math.ceil((self.sensitivity / self.epsilon) * math.log(2.0 / self.delta))
        
        # 2. 计算偏移中心 Center
        # Shift = k0 + Delta - 1
        self.shift = self.k0 + self.sensitivity - 1
        
        # 3. 计算截断上界 Upper Bound
        # Support in [0, 2 * shift]
        self.upper_bound = 2 * self.shift
        
        # 计算 Two-Sided Geometric 的参数 p
        # alpha = exp(-epsilon / Delta)
        # Numpy 的 geometric(p) 需要的是成功概率 p = 1 - alpha
        self.alpha = np.exp(-self.epsilon / self.sensitivity)
        self.geom_p = 1.0 - self.alpha

    def sample_two_sided_geometric(self):
        """
        采样双边几何分布 (Discrete Laplace)
        实现方式: Y = G1 - G2, where G1, G2 ~ Geometric(1-alpha)
        """
        g1 = np.random.geometric(self.geom_p)
        g2 = np.random.geometric(self.geom_p)
        return g1 - g2

    def generate_noise(self):
        """
        生成最终的 Shifted and Truncated Noise
        Result = min( max(0, Shift + TwoSidedGeom), UpperBound )
        """
        raw_geom = self.sample_two_sided_geometric()
        
        # 1. Shift
        shifted_val = self.shift + raw_geom
        
        # 2. Truncate (max 0)
        val_non_negative = max(0, shifted_val)
        
        # 3. Truncate (upper bound)
        final_val = min(val_non_negative, self.upper_bound)
        
        return final_val

class BucketProcessor:
    def __init__(self, partitions, epsilon, delta, sensitivity=1):
        """
        :param partitions: Partition 阶段输出的区间列表 [(start1, end1), (start2, end2), ...]
        :param epsilon: 用于 bucket padding 的隐私预算
        :param delta: 用于 bucket padding 的失败概率
        """
        self.partitions = partitions
        self.mechanism = ShiftedTruncatedGeometricMechanism(epsilon, delta, sensitivity)
        
    def distribute_and_pad(self, sorted_data, dummy_key=-1, dummy_freq=0):
        """
        将数据分发到各个 Bucket，并添加噪声填充 Dummy。
        
        :param sorted_data: 预处理好的 [(key, freq), ...]
        :return: List of Buckets, 每个 Bucket 是一个列表 [(key, freq), ..., (dummy, 0)]
        """
        # 1. 初始化 Buckets
        buckets = [[] for _ in range(len(self.partitions))]
        
        # 简单的线性扫描分发
        p_idx = 0
        num_partitions = len(self.partitions)
        
        for key, freq in sorted_data:
            while p_idx < num_partitions:
                start, end = self.partitions[p_idx]
                if key < start:
                    # Key 小于当前区间 (可能是 Gap 数据)，跳过
                    break 
                elif key > end:
                    # Key 超过当前区间，检查下一个区间
                    p_idx += 1
                else:
                    # Key 在当前区间内 [start, end]
                    buckets[p_idx].append((key, freq))
                    break
        
        # 2. 对每个 Bucket 进行 Padding
        final_buckets = []
        
        print(f"开始处理 {num_partitions} 个 Bucket (Adding Noise & Dummy)...")
        
        for i, real_items in enumerate(buckets):
            # A. 生成噪声 (决定需要填充多少 Dummy)
            noise_len = self.mechanism.generate_noise()
            
            # B. 填充 Dummy
            padding_needed = noise_len 
            
            padded_bucket = real_items.copy()
            dummies = [(dummy_key, dummy_freq)] * padding_needed
            padded_bucket.extend(dummies)
            
            # --- 修正点在此 ---
            final_buckets.append(padded_bucket)  # 之前写成了 padded_buckets
            
            # Debug info (打印前3个)
            if i < 3: 
                print(f"  Bucket {i} (Range {self.partitions[i]}): "
                      f"Real={len(real_items)}, Noise={noise_len}, Total={len(padded_bucket)}")
                
        return final_buckets