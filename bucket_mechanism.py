from noise_mechanisms import ShiftedTruncatedGeometricMechanism

class BucketProcessor:
    def __init__(self, partitions, epsilon, delta, sensitivity=1):
        """
        :param partitions: Partition 阶段输出的区间列表 [(start1, end1), (start2, end2), ...]
        :param epsilon: 用于 bucket padding 的隐私预算
        :param delta: 用于 bucket padding 的失败概率
        :param sensitivity: 敏感度 (默认为 1，即增加一行数据最多影响一个 bucket 的大小)
        """
        self.partitions = partitions
        
        # 1. 实例化外部的噪声机制
        # 这会自动计算 k0, shift, upper_bound 等参数
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
            # 直接调用外部机制生成非负整数噪声
            noise_len = self.mechanism.generate_noise()
            
            # B. 填充 Dummy
            # 在 Shifted Truncated 机制下，noise_len 本身就包含了 Shift (均值偏移)
            # 所以直接将其作为 padding 的数量是安全的
            padding_needed = noise_len 
            
            padded_bucket = real_items.copy()
            dummies = [(dummy_key, dummy_freq)] * padding_needed
            padded_bucket.extend(dummies)
            
            final_buckets.append(padded_bucket)
            
            # Debug info (打印前3个以确认逻辑)
            if i < 3: 
                print(f"  Bucket {i} (Range {self.partitions[i]}): "
                      f"Real={len(real_items)}, Noise={noise_len}, Total={len(padded_bucket)}")
                
        return final_buckets