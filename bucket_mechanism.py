from noise_mechanisms import ShiftedTruncatedGeometricMechanism

class BucketProcessor:
    def __init__(self, partitions, epsilon, delta, sensitivity=1):
        """
        :param partitions: Partition 阶段输出的区间列表
        :param epsilon: 用于所有 bucket padding 的总隐私预算
        :param delta: 用于所有 bucket padding 的总失败概率
        :param sensitivity: 桶内计数敏感度 (默认为 1，即增加一条记录，桶内计数+1)
        """
        self.partitions = partitions
        # 如果没有 partition，视为 1 个桶处理 Gap
        num_buckets = len(self.partitions) if self.partitions else 1
        
        # -------------------------------------------------------------
        # 【核心修正】并行组合 (Parallel Composition)
        # -------------------------------------------------------------
        # 之前的逻辑: self.delta_local = delta / max(1, num_buckets)
        # 现在的逻辑: 
        #由于 Partition 区间互不重叠，任意一条数据的变动仅影响 1 个 Bucket。
        # 根据差分隐私的并行组合定理，我们不需要在桶之间摊薄 Delta。
        # 每个 Bucket 都可以享受完整的隐私参数。
        self.delta_local = delta 
        
        # 实例化外部的噪声机制
        # 使用 Shifted Truncated 机制非常正确，因为它保证了非负整数噪声
        self.mechanism = ShiftedTruncatedGeometricMechanism(
            epsilon=epsilon, 
            delta=self.delta_local, 
            sensitivity=sensitivity
        )
        
        print(f"[BucketInit] Total Buckets: {num_buckets}")
        print(f"[BucketInit] Delta per bucket: {self.delta_local:.2e} (No division by B!)")
        print(f"[BucketInit] Noise Bound (k0): {self.mechanism.k0}, Shift: {self.mechanism.shift}")
        
    def distribute_and_pad(self, sorted_data, dummy_key=-1, dummy_freq=0):
        """
        将数据分发到各个 Bucket，并添加噪声填充 Dummy。
        
        :param sorted_data: 预处理好的 [(key, freq), ...]
        :return: List of Buckets, 每个 Bucket 是一个列表 [(key, freq), ..., (dummy, 0)]
        """
        # 1. 初始化 Buckets
        buckets = [[] for _ in range(len(self.partitions))]
        
        # 简单的线性扫描分发 (逻辑正确且高效，利用了 sorted_data 的有序性)
        p_idx = 0
        num_partitions = len(self.partitions)
        
        # 处理边界情况：如果没有 partitions，所有数据实际上都会被丢弃或放入默认桶
        # 这里假设 partitions 覆盖了感兴趣的范围
        if num_partitions > 0:
            for key, freq in sorted_data:
                while p_idx < num_partitions:
                    start, end = self.partitions[p_idx]
                    if key < start:
                        # Key 小于当前区间 (Gap 数据)，跳过
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
        
        # 打印日志稍微优化一下，只打印前几个
        print(f"开始处理 {num_partitions} 个 Bucket (Adding Noise & Dummy)...")
        
        for i, real_items in enumerate(buckets):
            # A. 生成噪声 (决定需要填充多少 Dummy)
            # 这里的噪声代表 "额外添加的虚假记录数量"
            # Shifted 机制保证了 noise_len >= 0 且均值已经偏移以掩盖真实数量
            noise_len = self.mechanism.generate_noise()
            
            # B. 填充 Dummy
            padding_needed = noise_len 
            
            padded_bucket = real_items.copy()
            # 只有当需要 padding 时才操作，微小的性能优化
            if padding_needed > 0:
                dummies = [(dummy_key, dummy_freq)] * padding_needed
                padded_bucket.extend(dummies)
            
            final_buckets.append(padded_bucket)
            
            # Debug info (打印前3个以确认逻辑)
            if i < 3: 
                print(f"  Bucket {i} (Range {self.partitions[i]}): "
                      f"Real={len(real_items)}, Noise={noise_len}, Total={len(padded_bucket)}")
                
        return final_buckets