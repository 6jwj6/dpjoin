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
        # return cls(a=true_mf,b=1)
    
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

    def run_join(self, parts_A, bucks_A, parts_B, bucks_B, meta_A=None, meta_B=None, dummy_key=-999, merge_factor=1):
        """
        流式区间对齐与本地 Join (保留双指针优化)
        新增了 Deterministic Upper Bound 截断：利用 meta_A.a 和 meta_B.a 斩断几何分布的噪声长尾！
        """
        joined_partitions = []
        joined_buckets = []
        
        i, j = 0, 0
        merged_items = []
        merged_start = None
        merged_end = None
        current_merge_count = 0
        
        # 记录当前合并批次中，真实的元素数量 (N_A 和 N_B)
        batch_size_A = 0
        batch_size_B = 0
        flag = 0
        while i < len(parts_A) and j < len(parts_B):
            start_A, end_A = parts_A[i]
            start_B, end_B = parts_B[j]
            
            intersect_start = max(start_A, start_B)
            intersect_end = min(end_A, end_B)
            
            if intersect_start <= intersect_end:
                b_A = bucks_A[i]
                b_B = bucks_B[j]
                
                valid_A = [item for item in b_A if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                valid_B = [item for item in b_B if item[0] == dummy_key or (intersect_start <= item[0] <= intersect_end)]
                
                # 累加当前批次的真实记录条数 (过滤掉 dummy)
                if flag == 1:
                    batch_size_A += sum(len(p_list) for k, f, p_list in b_A)
                elif flag == 2:
                    batch_size_B += sum(len(p_list) for k, f, p_list in b_B)
                elif flag == 0:
                    batch_size_A += sum(len(p_list) for k, f, p_list in b_A)
                    batch_size_B += sum(len(p_list) for k, f, p_list in b_B)
                
                payload_map_A = {}
                for k, f, p_list in valid_A:
                    if k != dummy_key:
                        payload_map_A.setdefault(k, []).extend(p_list)
                        
                payload_map_B = {}
                for k, f, p_list in valid_B:
                    if k != dummy_key:
                        payload_map_B.setdefault(k, []).extend(p_list)
                
                for k in payload_map_A:
                    if k in payload_map_B:
                        joined_payloads = []
                        for row_A in payload_map_A[k]:
                            for row_B in payload_map_B[k]:
                                joined_payloads.append({**row_A, **row_B})
                        
                        joined_freq = len(joined_payloads)
                        merged_items.append((k, joined_freq, joined_payloads))
                        
                if merged_start is None:
                    merged_start = intersect_start
                merged_end = intersect_end 
                current_merge_count += 1
                
                if current_merge_count >= merge_factor:
                    real_join_count = sum(len(p) for k, f, p in merged_items if k != dummy_key)
                    raw_noise = self.noise_mech.generate_noise()
                    
                    # ==========================================================
                    # 🚀 核心优化：Deterministic Upper Bound 截断
                    # ==========================================================
                    if meta_A and meta_B:
                        # 物理上限 = min(表A最高频次 * 表B的总条数, 表B最高频次 * 表A的总条数)
                        det_bound = min(meta_A.a * batch_size_B, meta_B.a * batch_size_A)
                        
                        if det_bound < real_join_count + raw_noise:
                            print(f"❌ Deterministic Upper Bound {det_bound}, noisy count{real_join_count + raw_noise}\n")
                        print(f"meta_A.a * batch_size_B, meta_B.a * batch_size_A: {meta_A.a}x{batch_size_B} , {meta_B.a}x{batch_size_A}")
                        # 最终目标大小取 DP加噪 和 物理上限 中的较小值
                        target_size = min(real_join_count + raw_noise, det_bound)
                        
                        # 倒推需要填充的 Dummy 数量
                        noise_len = max(0, int(target_size - real_join_count))
                    else:
                        noise_len = raw_noise
                    # ==========================================================
                        
                    if noise_len > 0:
                        merged_items.extend([(dummy_key, 0, [{'key':dummy_key}])] * noise_len)
                        
                    joined_partitions.append((merged_start, merged_end))
                    joined_buckets.append(merged_items)
                    
                    # 状态重置
                    merged_items = []
                    merged_start = None
                    merged_end = None
                    current_merge_count = 0
                    if end_A < end_B:
                        batch_size_A = 0
                    elif end_A > end_B:
                        batch_size_B = 0
                    else:
                        batch_size_A = 0
                        batch_size_B = 0
            
            # 双指针步进
            if end_A < end_B:
                i += 1
                flag = 1
            elif end_A > end_B:
                j += 1
                flag = 2
            else:
                i += 1
                j += 1
                flag = 0
                
        # 收尾处理 (逻辑同上)
        if current_merge_count > 0:
            real_join_count = sum(len(p) for k, f, p in merged_items if k != dummy_key)
            raw_noise = self.noise_mech.generate_noise()
            
            if meta_A and meta_B:
                det_bound = min(meta_A.a * batch_size_B, meta_B.a * batch_size_A)
                if det_bound < real_join_count + raw_noise:
                    print(f"❌ Deterministic Upper Bound {det_bound}, noisy count{real_join_count + raw_noise}\n")   
                print(f"meta_A.a * batch_size_B, meta_B.a * batch_size_A: {meta_A.a}x{batch_size_B} , {meta_B.a}x{batch_size_A}")
                target_size = min(real_join_count + raw_noise, det_bound)
                noise_len = max(0, int(target_size - real_join_count))
            else:
                noise_len = raw_noise
                
            if noise_len > 0:
                merged_items.extend([(dummy_key, 0, [{'key':dummy_key}])] * noise_len)
            
            joined_partitions.append((merged_start, merged_end))
            joined_buckets.append(merged_items)
                
        return joined_partitions, joined_buckets
    

    def run_full_join(self, parts_A, buckets_A, parts_B, buckets_B, meta_A=None, meta_B=None, dummy_key=-999, merge_factor=1, logger=None):
        """
        Bin-wise Full Join (Strict Oblivious Nested Loop + Deterministic Upper Bounding)
        保留区间双指针优化，但在桶内严格模拟 MPC 的 O(|b_A| * |b_B|) 算术电路开销。
        【修正】：绝对不进行任何预先的明文筛选（不提取 valid_A/B），确保对所有真实数据和 Dummy 的无差别扫描。
        """
        joined_partitions = []
        joined_buckets = []
        
        i, j = 0, 0
        
        print(f"开始执行跨表 Bin-wise Full Join (严格 Oblivious 桶内扫描 + DUB 截断优化)...")
        print(f"Merge Factor: {merge_factor}")
        
        merged_items = []
        merged_start = None
        merged_end = None
        current_merge_count = 0
        
        # 记录当前合并批次中，包含 Dummy 的物理元素总数
        batch_size_A = 0
        batch_size_B = 0
        flag = 0

        while i < len(parts_A) and j < len(parts_B):
            start_A, end_A = parts_A[i]
            start_B, end_B = parts_B[j]
            
            intersect_start = max(start_A, start_B)
            intersect_end = min(end_A, end_B)
            
            if intersect_start <= intersect_end:
                b_A = buckets_A[i]
                b_B = buckets_B[j]
                
                # 累加当前批次的物理记录总数 (不去除 Dummy)，用于确定性截断 (DUB)
                if flag == 1:
                    batch_size_A += sum(len(p_list) for k, f, p_list in b_A)
                elif flag == 2:
                    batch_size_B += sum(len(p_list) for k, f, p_list in b_B)
                elif flag == 0:
                    batch_size_A += sum(len(p_list) for k, f, p_list in b_A)
                    batch_size_B += sum(len(p_list) for k, f, p_list in b_B)

                # =======================================================
                # 🚀 新增：记录当前交叉的两个 Bucket 的真实物理行数并打点
                # =======================================================
                if logger:
                    rows_A = sum(len(p_list) for k, f, p_list in b_A)
                    rows_B = sum(len(p_list) for k, f, p_list in b_B)
                    logger.log_bucket_pair(rows_A, rows_B)
                # =======================================================

                temp_merged_dict = {}
                
                # ---------------------------------------------------
                # 【Oblivious 核心模拟】：绝对全量遍历，无差别矩阵扫描
                # 不做任何明文筛选，拿整个桶 b_A 和 b_B 直接硬算！
                # ---------------------------------------------------
                for k_A, f_A, p_list_A in b_A:
                    for k_B, f_B, p_list_B in b_B:
                        
                        # MPC 电路判定：Key 相等、非 Dummy，且必须都落在当前的交叉区间内
                        # in_range_A = (intersect_start <= k_A <= intersect_end)
                        # in_range_B = (intersect_start <= k_B <= intersect_end)
                        
                        # is_match = (k_A == k_B) and (k_A != dummy_key) and in_range_A and in_range_B
                        is_match = (k_A == k_B) and (k_A != dummy_key)
                        
                        # 遍历载荷列表（因为 Dummy 的 p_list 是 [{}]，所有元素都会无差别执行这里的循环）
                        for row_A in p_list_A:
                            for row_B in p_list_B:
                                
                                # 强制执行字典拼接！模拟底层同态加法/乘法开销
                                merged_row = {**row_A, **row_B}
                                
                                # 模拟 MUX (多路选择器)：只有电路输出为 1 时，才将结果“写入”有效缓冲区
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
                    # 统计目前产生了多少条真实 Join 结果
                    real_join_count = sum(len(p) for k, f, p in merged_items if k != dummy_key)
                    raw_noise = self.noise_mech.generate_noise()
                    
                    # ==========================================================
                    # 🚀 Deterministic Upper Bound 截断
                    # ==========================================================
                    if meta_A and meta_B:
                        # 绝对物理上限计算
                        det_bound = min(meta_A.a * batch_size_B, meta_B.a * batch_size_A)
                        if det_bound < real_join_count + raw_noise:
                            print(f"❌ Deterministic Upper Bound {det_bound}, noisy count {real_join_count + raw_noise}")
                        print(f"meta_A.a * batch_size_B, meta_B.a * batch_size_A: {meta_A.a}x{batch_size_B} , {meta_B.a}x{batch_size_A}")
                        
                        # 目标大小取两者中较小的一个
                        target_size = min(real_join_count + raw_noise, det_bound)
                        
                        # 反推最终需要的噪声填充量，确保非负
                        noise_len = max(0, int(target_size - real_join_count))
                    else:
                        noise_len = raw_noise
                    # ==========================================================
                        
                    if noise_len > 0:
                        # 统一 Dummy 结构
                        merged_items.extend([(dummy_key, 0, [{'key':dummy_key}])] * noise_len)
                        
                    joined_partitions.append((merged_start, merged_end))
                    joined_buckets.append(merged_items)
                    
                    # 状态重置
                    merged_items = []
                    merged_start = None
                    merged_end = None
                    current_merge_count = 0
                    if end_A < end_B:
                        batch_size_A = 0
                    elif end_A > end_B:
                        batch_size_B = 0
                    else:
                        batch_size_A = 0
                        batch_size_B = 0
            
            # 双指针步进
            if end_A < end_B:
                i += 1
                flag = 1
            elif end_A > end_B:
                j += 1
                flag = 2
            else:
                i += 1
                j += 1
                flag = 0
                
        # 收尾处理
        if current_merge_count > 0:
            real_join_count = sum(len(p) for k, f, p in merged_items if k != dummy_key)
            raw_noise = self.noise_mech.generate_noise()
            
            if meta_A and meta_B:
                det_bound = min(meta_A.a * batch_size_B, meta_B.a * batch_size_A)
                if det_bound < real_join_count + raw_noise:
                    print(f"❌ Deterministic Upper Bound {det_bound}, noisy count {real_join_count + raw_noise}")
                print(f"meta_A.a * batch_size_B, meta_B.a * batch_size_A: {meta_A.a}x{batch_size_B} , {meta_B.a}x{batch_size_A}")
                target_size = min(real_join_count + raw_noise, det_bound)
                noise_len = max(0, int(target_size - real_join_count))
            else:
                noise_len = raw_noise
                
            if noise_len > 0:
                merged_items.extend([(dummy_key, 0, [{'key':dummy_key}])] * noise_len)
            
            joined_partitions.append((merged_start, merged_end))
            joined_buckets.append(merged_items)
                
        print(f"Oblivious Bin-wise Full Join 完成！实际生成了 {len(joined_partitions)} 个物理 Bucket。")
        return joined_partitions, joined_buckets