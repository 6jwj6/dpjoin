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
    
    @classmethod
    def from_base_table_multiple_keys(cls, list_of_key_lists, epsilon, delta):
        """
        [外键桥接表专用]
        计算包含多个潜在 Join Key 的表的频次上限。
        在明文下取多个属性频次的最大值，然后统一加一次噪声，节省隐私预算！
        """
        if not list_of_key_lists or not all(list_of_key_lists):
            return cls(1, 1)
        
        from collections import Counter
        
        # 1. 在明文下找到所有关联键中的最大真实频次
        max_true_mf = 1
        for keys in list_of_key_lists:
            if keys:
                current_mf = max(Counter(keys).values())
                max_true_mf = max(max_true_mf, current_mf)
                
        # 2. 统一添加一次噪声 (整体敏感度依然为 1)
        mech = ShiftedTruncatedGeometricMechanism(epsilon=epsilon, delta=delta, sensitivity=1)
        raw_noise = mech.sample_two_sided_geometric()
        
        # 3. 计算高置信度上限
        noisy_mf = max(1, int(max_true_mf + raw_noise + mech.k0))
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
        (支持带有 Payload 的笛卡尔积连接)
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
                
                # 边界过滤 (兼容新的三元组结构)
                valid_A = [item for item in b_A if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                valid_B = [item for item in b_B if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                
                # ---------------------------------------------------
                # 核心升级 1：提取携带 Payload 的映射
                # ---------------------------------------------------
                payload_map_A = {}
                for k, f, p_list in valid_A:
                    if k != dummy_key:
                        payload_map_A.setdefault(k, []).extend(p_list)
                        
                payload_map_B = {}
                for k, f, p_list in valid_B:
                    if k != dummy_key:
                        payload_map_B.setdefault(k, []).extend(p_list)
                
                # ---------------------------------------------------
                # 核心升级 2：笛卡尔积与属性字典合并
                # ---------------------------------------------------
                for k in payload_map_A:
                    if k in payload_map_B:
                        joined_payloads = []
                        # 遍历 A 和 B 中所有匹配该 Key 的行，执行全连接
                        for row_A in payload_map_A[k]:
                            for row_B in payload_map_B[k]:
                                # Python 字典合并操作：合并 A 和 B 的属性
                                merged_row = {**row_A, **row_B}
                                joined_payloads.append(merged_row)
                        
                        joined_freq = len(joined_payloads)
                        # 将合并后的 payloads 放回结果中
                        merged_items.append((k, joined_freq, joined_payloads))
                        
                # ---------------------------------------------------
                # 缓冲区逻辑保持不变，但 Dummy 数据结构升级
                # ---------------------------------------------------
                if merged_start is None:
                    merged_start = intersect_start
                merged_end = intersect_end 
                current_merge_count += 1
                
                if current_merge_count >= merge_factor:
                    noise_len = self.noise_mech.generate_noise()
                    
                    if noise_len > 0:
                        # 核心升级 3：Dummy 携带空的 payload 列表
                        dummies = [(dummy_key, 0, [])] * noise_len
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
                merged_items.extend([(dummy_key, 0, [])] * noise_len)
            
            joined_partitions.append((merged_start, merged_end))
            joined_buckets.append(merged_items)
                
        print(f"Join 完成！通过合并优化，实际生成了 {len(joined_partitions)} 个物理 Bucket。")
        return joined_partitions, joined_buckets
    
    def run_full_join(self, parts_A, buckets_A, parts_B, buckets_B, dummy_key=-999, merge_factor=1):
        """
        Bin-wise Full Join (Strict Oblivious Nested Loop)
        保留区间双指针优化，但在桶内严格模拟 MPC 的 O(N * M) 算术电路开销。
        不跳过 Dummy 匹配，确保所有载荷都参与运算以实现时间上的不可区分性。
        """
        joined_partitions = []
        joined_buckets = []
        
        i, j = 0, 0
        
        print(f"开始执行跨表 Bin-wise Full Join (严格 Oblivious 桶内扫描)...")
        print(f"Merge Factor: {merge_factor}")
        
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
                
                # 提取数据（注意：这里已经包含了所有的 Dummy 数据）
                valid_A = [item for item in b_A if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                valid_B = [item for item in b_B if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                
                temp_merged_dict = {}
                
                # ---------------------------------------------------
                # 【Oblivious 核心模拟】：全量遍历，不跳过 Dummy
                # ---------------------------------------------------
                for k_A, f_A, p_list_A in valid_A:
                    for k_B, f_B, p_list_B in valid_B:
                        
                        # MPC 电路中的判定条件：Key 相等且均不是 Dummy
                        is_match = (k_A == k_B) and (k_A != dummy_key)
                        
                        # 遍历载荷列表（因为 Dummy 的 p_list 是 [{}]，这里同样会执行）
                        for row_A in p_list_A:
                            for row_B in p_list_B:
                                
                                # 强制执行字典拼接！
                                # 这模拟了底层不管 is_match 为何值，都要进行的同态加法/乘法开销
                                merged_row = {**row_A, **row_B}
                                
                                # 模拟 MUX (多路选择器)：只有匹配时，才将结果“写入”有效缓冲区
                                if is_match:
                                    temp_merged_dict.setdefault(k_A, []).append(merged_row)
                
                for k, p_list in temp_merged_dict.items():
                    merged_items.append((k, len(p_list), p_list))
                # ---------------------------------------------------
                        
                if merged_start is None:
                    merged_start = intersect_start
                merged_end = intersect_end 
                current_merge_count += 1
                
                if current_merge_count >= merge_factor:
                    noise_len = self.noise_mech.generate_noise()
                    if noise_len > 0:
                        # 【核心修正】：第三项替换为 [{}]，使 Dummy 结构和真实数据一致
                        merged_items.extend([(dummy_key, 0, [{}])] * noise_len)
                        
                    joined_partitions.append((merged_start, merged_end))
                    joined_buckets.append(merged_items)
                    
                    merged_items = []
                    merged_start = None
                    merged_end = None
                    current_merge_count = 0
            
            if end_A < end_B:
                i += 1
            elif end_A > end_B:
                j += 1
            else:
                i += 1
                j += 1
                
        # 收尾处理
        if current_merge_count > 0:
            noise_len = self.noise_mech.generate_noise()
            if noise_len > 0:
                merged_items.extend([(dummy_key, 0, [{}])] * noise_len)
            
            joined_partitions.append((merged_start, merged_end))
            joined_buckets.append(merged_items)
                
        print(f"Bin-wise Full Join 完成！实际生成了 {len(joined_partitions)} 个物理 Bucket。")
        return joined_partitions, joined_buckets