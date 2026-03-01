import numpy as np
from noise_mechanisms import ShiftedTruncatedGeometricMechanism

class DPJoiner:
    def __init__(self, epsilon: float, delta: float, sensitivity: int = 1):
        """
        初始化 DP Joiner
        
        Args:
            epsilon: 分配给 Join 阶段加噪的隐私预算
            delta: 分配给 Join 阶段的失败概率
            sensitivity: Join 结果数量的敏感度。
                         注意：在 Join 操作中，如果表 A 增加一条记录，Join 结果的增加量
                         取决于该 Key 在表 B 中的频次。因此这里的 sensitivity 通常是
                         受限于某个截断阈值 (Max Frequency Bound)。
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Join 生成的结果本质上是记录条数，所以使用整数噪声机制
        self.noise_mech = ShiftedTruncatedGeometricMechanism(
            epsilon=self.epsilon, 
            delta=self.delta, 
            sensitivity=self.sensitivity
        )
        
        print(f"[JoinInit] Epsilon={self.epsilon}, Delta={self.delta}, Sensitivity={self.sensitivity}")
        print(f"[JoinInit] Noise Shift={self.noise_mech.shift}, Max Noise={self.noise_mech.upper_bound}")

    def run_join(self, parts_A, buckets_A, parts_B, buckets_B, dummy_key=-999):
        """
        执行带有对齐、边界处理、加噪和 Compact 的 Join 操作
        
        返回:
            joined_partitions: 交叉对齐后的新区间列表
            joined_buckets: 对应新区间的结果 Buckets (包含 Real + Dummy 数据)
        """
        joined_partitions = []
        joined_buckets = []
        
        # 使用双指针法扫描并对齐两个排序好的 Partitions
        i, j = 0, 0
        
        print("开始执行跨表 Interval Alignment & Local Join...")
        
        while i < len(parts_A) and j < len(parts_B):
            start_A, end_A = parts_A[i]
            start_B, end_B = parts_B[j]
            
            # 1. 确定交叉区间 (Intersection Segment)
            intersect_start = max(start_A, start_B)
            intersect_end = min(end_A, end_B)
            
            if intersect_start <= intersect_end:
                # 存在有效交叉区域
                joined_partitions.append((intersect_start, intersect_end))
                
                b_A = buckets_A[i]
                b_B = buckets_B[j]
                
                # ---------------------------------------------------
                # 2. 处理边界值 (Boundary Handling)
                # ---------------------------------------------------
                # 过滤出落在交叉区间内的真实数据，丢弃越界数据。
                # 此时 Dummy 数据 (key == dummy_key) 不受边界限制，暂时保留参与逻辑
                valid_A = [item for item in b_A if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                valid_B = [item for item in b_B if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                
                # ---------------------------------------------------
                # 3. 本地 Join 执行 (Local Join)
                # ---------------------------------------------------
                # 为了模拟，我们将其转化为 Dict 以实现 O(N) 的 Equi-Join
                freq_map_A = {}
                for k, f in valid_A:
                    if k != dummy_key:
                        freq_map_A[k] = freq_map_A.get(k, 0) + f
                        
                freq_map_B = {}
                for k, f in valid_B:
                    if k != dummy_key:
                        freq_map_B[k] = freq_map_B.get(k, 0) + f
                
                joined_items = []
                # 遍历执行频率相乘 (频次 Join 的数学等价)
                for k in freq_map_A:
                    if k in freq_map_B:
                        joined_freq = freq_map_A[k] * freq_map_B[k]
                        joined_items.append((k, joined_freq))
                        
                # ---------------------------------------------------
                # 4. 加噪与 Compact (Resize & Padding)
                # ---------------------------------------------------
                # 在真实的 MPC 中，这里会涉及 obliviously 丢弃 dummy，
                # 然后对剩余的真实结果进行 Oblivious Compaction 到一个目标大小。
                # 在模拟器中，目标大小 = len(真实 Join 结果) + DP 噪声
                
                noise_len = self.noise_mech.generate_noise()
                
                # 构建最终的 Compact Bucket
                compact_bucket = joined_items.copy()
                
                # 填充 Dummy 数据直至达到 Target Size
                if noise_len > 0:
                    dummies = [(dummy_key, 0)] * noise_len
                    compact_bucket.extend(dummies)
                    
                joined_buckets.append(compact_bucket)
            
            # 移动指针：谁的结束端点更靠左，谁就向前移动寻找下一个潜在交叉
            if end_A < end_B:
                i += 1
            else:
                j += 1
                
        print(f"Join 完成！生成了 {len(joined_partitions)} 个交叉分段。")
        return joined_partitions, joined_buckets