import numpy as np
from collections import Counter
from noise_mechanisms import ShiftedTruncatedGeometricMechanism

class JoinMetadata:
    """
    用于在多表 Join 过程中追踪和计算加噪最大频次 (a) 和敏感度 (b)
    """
    def __init__(self, a: int, b: int):
        self.a = a  # 频次上限 (Max Frequency / mf)
        self.b = b  # 当前状态的敏感度 (Sensitivity)

    @classmethod
    def from_base_table(cls, table_keys, epsilon, delta):
        """
        模拟数据拥有者本地的保护操作。
        这里生成的 a 被认为是一个绝对安全的上限，因此后续 Join 无需再次物理截断。
        基础表的敏感度 b 恒为 1。
        """
        if len(table_keys) == 0:
            return cls(1, 1)
        
        # 1. 获取真实的最大频次 (Integer)
        true_mf = max(Counter(table_keys).values())
        
        # 2. 统一使用已有的 ShiftedTruncatedGeometricMechanism
        # 基础表最大频次的敏感度为 1 (多一条数据，某个 Key 的频次最多 +1)
        mech = ShiftedTruncatedGeometricMechanism(epsilon=epsilon, delta=delta, sensitivity=1)
        
        # 3. 采样纯净的双边几何噪声 (Discrete Laplace)
        raw_noise = mech.sample_two_sided_geometric()
        
        # 4. 计算安全的加噪上限：真实频次 + 随机噪声 + 安全余量(k0)
        # 这里的 mech.k0 正好等于之前公式里的 margin = scale * ln(1/delta) 在离散域上的映射
        noisy_mf = true_mf + raw_noise + mech.k0
        
        # 5. 后处理 (Post-processing)：频次至少为 1
        noisy_mf = max(1, int(noisy_mf))
        
        return cls(a=noisy_mf, b=1)

    def join(self, other: 'JoinMetadata') -> 'JoinMetadata':
        """
        严格按照敏感度传播公式更新 a 和 b
        a = a1 * a2
        b = max(a1 * b2, a2 * b1)
        """
        new_a = self.a * other.a
        new_b = max(self.a * other.b, other.a * self.b)
        return JoinMetadata(a=new_a, b=new_b)


class DPJoiner:
    def __init__(self, epsilon: float, delta: float, sensitivity: int = 1):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        self.noise_mech = ShiftedTruncatedGeometricMechanism(
            epsilon=self.epsilon, 
            delta=self.delta, 
            sensitivity=self.sensitivity
        )
        
        print(f"[JoinInit] Epsilon={self.epsilon}, Delta={self.delta}, Sensitivity={self.sensitivity}")
        print(f"[JoinInit] Noise Shift={self.noise_mech.shift}, Max Noise={self.noise_mech.upper_bound}")

    def run_join(self, parts_A, buckets_A, parts_B, buckets_B, dummy_key=-999, merge_factor=1):
        """
        执行带有区间对齐、本地 Join、相邻桶合并优化以及加噪 Compact 的 Join 操作。
        (已移除物理截断逻辑，信任源数据频次不越界)
        """
        joined_partitions = []
        joined_buckets = []
        
        i, j = 0, 0
        
        print(f"开始执行跨表 Interval Alignment & Local Join (Merge Factor: {merge_factor})...")
        
        merged_items = []
        merged_start = None
        merged_end = None
        current_merge_count = 0
        
        while i < len(parts_A) and j < len(parts_B):
            start_A, end_A = parts_A[i]
            start_B, end_B = parts_B[j]
            
            intersect_start = max(start_A, start_B)
            intersect_end = min(end_A, end_B)
            
            if intersect_start <= intersect_end:
                b_A = buckets_A[i]
                b_B = buckets_B[j]
                
                # 边界过滤
                valid_A = [item for item in b_A if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                valid_B = [item for item in b_B if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                
                # 本地 Join 频率计算 (不进行截断)
                freq_map_A = {}
                for k, f in valid_A:
                    if k != dummy_key:
                        freq_map_A[k] = freq_map_A.get(k, 0) + f
                        
                freq_map_B = {}
                for k, f in valid_B:
                    if k != dummy_key:
                        freq_map_B[k] = freq_map_B.get(k, 0) + f
                
                # 执行交叉区间内的等值连接
                for k in freq_map_A:
                    if k in freq_map_B:
                        joined_freq = freq_map_A[k] * freq_map_B[k]
                        merged_items.append((k, joined_freq))
                        
                # 合并逻辑：更新当前聚合区间的边界范围
                if merged_start is None:
                    merged_start = intersect_start
                merged_end = intersect_end 
                current_merge_count += 1
                
                # 当达到指定的合并数量时，执行一次统一的加噪与封装
                if current_merge_count >= merge_factor:
                    noise_len = self.noise_mech.generate_noise()
                    
                    if noise_len > 0:
                        dummies = [(dummy_key, 0)] * noise_len
                        merged_items.extend(dummies)
                        
                    joined_partitions.append((merged_start, merged_end))
                    joined_buckets.append(merged_items)
                    
                    merged_items = []
                    merged_start = None
                    merged_end = None
                    current_merge_count = 0
            
            if end_A < end_B:
                i += 1
            else:
                j += 1
                
        # 收尾处理
        if current_merge_count > 0:
            noise_len = self.noise_mech.generate_noise()
            if noise_len > 0:
                merged_items.extend([(dummy_key, 0)] * noise_len)
            
            joined_partitions.append((merged_start, merged_end))
            joined_buckets.append(merged_items)
                
        print(f"Join 完成！通过合并优化，实际生成了 {len(joined_partitions)} 个物理 Bucket。")
        return joined_partitions, joined_buckets