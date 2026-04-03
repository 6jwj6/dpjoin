import numpy as np
from noise_mechanisms import StandardLaplaceMechanism

class PrivatePartitionOffline:
    def __init__(self, epsilon, delta, domain_size, sensitivity=1):
        """
        初始化 Partition 算法 (Offline / Gap-Jumping Version)
        
        【核心理论】
        此算法利用 Offline 特性加速：在 Count=0 的区域直接跳过。
        为了保证这种"跳过行为"在隐私上是安全的（即与 Online 版本不可区分），
        我们需要设定一个足够高的阈值 T。
        
        Args:
            epsilon: 隐私预算 epsilon (用于控制噪声规模)
            delta:   隐私预算 delta (这里显式消耗 Delta，用于换取 Offline 执行的权利)
                     我们必须保证：Pr[Gap中任意一点噪声 > T] <= delta
            domain_size: 域大小 D (用于 Union Bound)
            sensitivity: 敏感度 (默认为 1)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.sensitivity = sensitivity
        
        # 1. 实例化标准拉普拉斯机制 (无截断，范围 -inf 到 +inf)
        self.laplace_mech = StandardLaplaceMechanism(
            epsilon=self.epsilon, 
            sensitivity=self.sensitivity
        )
        self.scale = self.laplace_mech.scale
        
        # 2. 计算隐私阈值 T (Privacy Threshold)
        safe_delta = min(self.delta, 0.5) # 防止数值错误
        self.T = (2 * self.scale) * np.log(self.D / (2 * safe_delta))
        
        print(f"[Init] Offline Partition (Standard Laplace)")
        print(f"[Init] Epsilon={self.epsilon}, Delta(Privacy)={self.delta}")
        print(f"[Init] Privacy Threshold (T)={self.T:.4f}")

    def get_seal_probability(self, current_load, z_init):
        """
        计算在当前状态下发生密封(Seal)的概率。
        """
        distance = self.T + z_init - current_load
        
        if distance >= 0:
            p = 0.5 * np.exp(-distance / self.scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / self.scale)
            
        return np.clip(p, 0.0, 1.0)

    def preprocess_data(self, raw_keys, raw_payloads):
        """
        [升维改造] 
        将 keys 和 payloads 组合，聚合相同 key 的 payloads。
        返回格式: [(key, freq, payload_list), ...]
        """
        from collections import defaultdict
        if len(raw_keys) == 0: return []
        
        data_map = defaultdict(list)
        for k, p in zip(raw_keys, raw_payloads):
            data_map[k].append(p)
            
        # 根据 Key 排序，并打包为 (key, 频次, 载荷列表)
        sorted_data = [(k, len(p_list), p_list) for k, p_list in sorted(data_map.items(), key=lambda x: x[0])]
        return sorted_data

    def run_partition(self, sorted_data):
        output_splits = []
        current_load = 0
        
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        last_pos = 0 
        
        # [升维改造] 解包时接收 payload_list (这里用不到 payload，所以用 _ 占位)
        for key, freq, _ in sorted_data:
            # =================================================
            # Part 1: Gap 处理 (The Offline Optimization)
            # =================================================
            gap_len = key - last_pos - 1
            
            if gap_len > 0:
                p = self.get_seal_probability(current_load, z_init)
                
                if p > 1e-15:
                    try:
                        steps = np.random.geometric(p)
                        
                        if steps <= gap_len:
                            cut_pos = last_pos + steps
                            output_splits.append(int(cut_pos))
                            
                            current_load = 0
                            z_init = self.laplace_mech.generate_noise()
                            target = self.T + z_init
                    except ValueError:
                        pass 
            
            # =================================================
            # Part 2: Key 处理 (Standard Online Logic)
            # =================================================
            current_load += freq
            
            z_check = self.laplace_mech.generate_noise()
            
            if current_load + z_check > target:
                output_splits.append(int(key))
                
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            last_pos = key

        # =================================================
        # Part 3: Final Gap 处理
        # =================================================
        final_gap = self.D - last_pos
        if final_gap > 0:
            p = self.get_seal_probability(current_load, z_init)
            if p > 1e-15:
                try:
                    steps = np.random.geometric(p)
                    if steps <= final_gap:
                        output_splits.append(int(last_pos + steps))
                except:
                    pass

        return self._format_output(output_splits)

    def _format_output(self, output_splits):
        valid_splits = sorted([s for s in output_splits if 1 <= s <= self.D])
        final_partitions = []
        start = 1
        for split in valid_splits:
            if start <= split:
                final_partitions.append((int(start), int(split)))
                start = split + 1
        if start <= self.D:
            final_partitions.append((int(start), int(self.D)))
        return final_partitions
    


from concurrent.futures import ProcessPoolExecutor # [优化1] 换用多进程
from collections import Counter

class PrivatePartitionParallel:
    def __init__(self, epsilon, delta, domain_size, num_threads=10, sensitivity=1):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.num_threads = num_threads
        self.sensitivity = sensitivity
        
        self.eps_chunk = self.epsilon * 0.1
        self.eps_seq = self.epsilon * 0.9
        
        self.laplace_mech = StandardLaplaceMechanism(
            epsilon=self.eps_seq, 
            sensitivity=self.sensitivity
        )
        self.scale = self.laplace_mech.scale
        
        safe_delta = min(self.delta, 0.5) 
        self.T = (2 * self.scale) * np.log(self.D / (2 * safe_delta))
        
        print(f"[Init] True Distributed Partition - Workers: {self.num_threads}")

    def get_seal_probability(self, current_load, z_init):
        distance = self.T + z_init - current_load
        if distance >= 0:
            return min(1.0, max(0.0, 0.5 * np.exp(-distance / self.scale)))
        else:
            return min(1.0, max(0.0, 1.0 - 0.5 * np.exp(distance / self.scale)))

    def _optimize_worker_count(self, N):
        """
        自适应并发数约束。
        注意：因为输入是扁平数组，N 本身就代表了总数据量 (total_freq)。
        """
        K = self.num_threads
        if K <= 1:
            return 1

        original_K = K

        # 约束一：阈值饥饿 (确保每个线程分到的原始数据量至少是目标阈值 T 的 3 倍)
        if (N / K) < 3 * self.T:
            K_new = max(1, int(N / (3 * self.T)))
            if K_new < K:
                K = K_new
                
        # 约束二：切分噪声吞噬 (确保每个物理块大小至少是噪声 Scale 的 10 倍)
        noise_scale = self.sensitivity / self.eps_chunk
        if (N / K) < 10 * noise_scale:
            K_new = max(1, int(N / (10 * noise_scale)))
            if K_new < K:
                K = K_new
                
        # 约束三：系统开销 (数据太少不值得多进程，直接单核)
        if N < 50_000:
            K = 1

        if K < original_K:
            print(f"  [系统提示] 触发安全约束，并发数自适应调整: {original_K} -> {K}")

        return K

    def _local_partition(self, raw_chunk):
        """
        [Worker 核心] 
        接收原始的局部数据段，在线程内独立完成聚合、排序和 SVT 切分。
        """
        output_splits = []
        if not len(raw_chunk):
            return output_splits
            
        # 1. 线程内局部聚合 (Local GroupBy & Sort)
        # 这是真正的分布式精髓：把 O(N log N) 的压力分散给各个 CPU
        local_counts = Counter(raw_chunk)
        sorted_local_data = sorted(local_counts.items(), key=lambda x: x[0])
            
        current_load = 0
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        last_pos = sorted_local_data[0][0] - 1 
        
        # 2. 局部 Gap-Truncated SVT 遍历
        for key, freq in sorted_local_data:
            # --- Part 1: Gap 处理 ---
            gap_len = key - last_pos - 1
            if gap_len > 0:
                p = self.get_seal_probability(current_load, z_init)
                if p > 1e-15:
                    try:
                        steps = np.random.geometric(p)
                        if steps <= gap_len:
                            cut_pos = last_pos + steps
                            output_splits.append(int(cut_pos))
                            
                            current_load = 0
                            z_init = self.laplace_mech.generate_noise()
                            target = self.T + z_init
                    except ValueError:
                        pass 
            
            # --- Part 2: 实际 Key 处理 ---
            current_load += freq
            z_check = self.laplace_mech.generate_noise()
            
            if current_load + z_check > target:
                output_splits.append(int(key))
                
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            last_pos = key

        # 3. 跑完局部数据，直接停机上报
        return output_splits

    def run_partition(self, raw_keys):
        """
        [Master 主控]
        只负责极轻量级的、带噪的物理 Index 数组切割，并将切片发送给进程池。
        """
        N = len(raw_keys)
        if N == 0:
            return self._format_output([self.D])
            
        # 获取自适应安全并发数
        K = self._optimize_worker_count(N)
        
        if K <= 1:
            splits = self._local_partition(raw_keys)
            splits.append(self.D)
            return self._format_output(sorted(list(set(splits))))

        # --- Stage 1: 带噪物理切分 (只切分原始数组) ---
        base_size = N / K
        indices = [0]
        current_idx = 0
        
        for i in range(K - 1):
            noise = np.random.laplace(0, self.sensitivity / self.eps_chunk)
            chunk_size = int(round(base_size + noise))
            chunk_size = max(1, chunk_size)
            current_idx = min(current_idx + chunk_size, N)
            indices.append(current_idx)
            
        indices.append(N)
        for i in range(1, len(indices)):
            if indices[i] < indices[i-1]:
                indices[i] = indices[i-1]

        # --- Stage 2: 真正的数据并行下发 ---
        global_splits = []
        futures = []
        
        with ProcessPoolExecutor(max_workers=K) as executor:
            for i in range(K):
                start_idx = indices[i]
                end_idx = indices[i+1]
                if start_idx == end_idx:
                    continue 
                    
                # 提取局部的扁平数组 (完全没有额外的 payload 或 元组包装，传输开销极小)
                chunk_raw_data = raw_keys[start_idx:end_idx]
                
                futures.append(executor.submit(self._local_partition, chunk_raw_data))

        # --- Stage 3: 无脑拼接与去重 ---
        for future in futures:
            global_splits.extend(future.result())
            
        global_splits.append(self.D)
        global_splits = sorted(list(set(global_splits)))
        
        return self._format_output(global_splits)

    def _format_output(self, output_splits):
        valid_splits = [s for s in output_splits if 1 <= s <= self.D]
        final_partitions = []
        start = 1
        for split in valid_splits:
            if start <= split:
                final_partitions.append((int(start), int(split)))
                start = split + 1
        if start <= self.D:
            final_partitions.append((int(start), int(self.D)))
        return final_partitions