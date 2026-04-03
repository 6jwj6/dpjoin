import numpy as np
from noise_mechanisms import StandardLaplaceMechanism

class PrivatePartitionOffline:
    def __init__(self, epsilon, delta, domain_size, sensitivity=1):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.sensitivity = sensitivity
        
        self.laplace_mech = StandardLaplaceMechanism(
            epsilon=self.epsilon, 
            sensitivity=self.sensitivity
        )
        self.scale = self.laplace_mech.scale
        
        safe_delta = min(self.delta, 0.5) 
        self.T = (2 * self.scale) * np.log(self.D / (2 * safe_delta))
        
        print(f"[Init] Streaming Offline Partition")
        print(f"[Init] T={self.T:.4f}, Scale={self.scale:.4f}")

    def get_seal_probability(self, current_load, z_init):
        distance = self.T + z_init - current_load
        if distance >= 0:
            p = 0.5 * np.exp(-distance / self.scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / self.scale)
        return np.clip(p, 0.0, 1.0)

    def run_partition(self, raw_keys):
        """
        [流式优化版] O(N) 扫描，无需聚合字典。
        前提：raw_keys 必须已经从小到大排序。
        """
        output_splits = []
        if not len(raw_keys):
            return self._format_output([self.D])
            
        current_load = 0
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        last_processed_key = 0 # 记录上一个处理完的逻辑位置
        
        # 记录当前正在扫描的 Key 及其频率
        active_key = raw_keys[0]
        active_freq = 0
        
        def process_key_event(key, freq, last_pos):
            """内部辅助：处理一个 (Key, Freq) 组合的 SVT 逻辑"""
            nonlocal current_load, z_init, target
            
            # --- Part 1: Gap 处理 (从上个 Key 到当前 Key 之间的空白) ---
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
                    except: pass
            
            # --- Part 2: Key 处理 ---
            current_load += freq
            z_check = self.laplace_mech.generate_noise()
            if current_load + z_check > target:
                output_splits.append(int(key))
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            return key # 返回更新后的 last_pos

        # --- 核心流式扫描 ---
        for k in raw_keys:
            if k == active_key:
                active_freq += 1
            else:
                # 发现 Key 变化，处理前一个 Key 的积压数据
                last_processed_key = process_key_event(active_key, active_freq, last_processed_key)
                # 重置 active 状态
                active_key = k
                active_freq = 1
        
        # 循环结束，处理最后一个 Key
        last_processed_key = process_key_event(active_key, active_freq, last_processed_key)

        # --- Part 3: Final Gap 处理 (直到 D) ---
        final_gap = self.D - last_processed_key
        if final_gap > 0:
            p = self.get_seal_probability(current_load, z_init)
            if p > 1e-15:
                try:
                    steps = np.random.geometric(p)
                    if steps <= final_gap:
                        output_splits.append(int(last_processed_key + steps))
                except: pass

        return self._format_output(output_splits)

    def _format_output(self, output_splits):
        valid_splits = sorted(list(set([s for s in output_splits if 1 <= s <= self.D])))
        final_partitions = []
        start = 1
        for split in valid_splits:
            if start <= split:
                final_partitions.append((int(start), int(split)))
                start = split + 1
        if start <= self.D:
            final_partitions.append((int(start), int(self.D)))
        return final_partitions



from concurrent.futures import ProcessPoolExecutor

class PrivatePartitionParallel:
    def __init__(self, epsilon, delta, domain_size, num_threads=10, sensitivity=1):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.num_threads = num_threads
        self.sensitivity = sensitivity
        
        # 预算拆分：10% 用于物理块切割，90% 用于实际 SVT 计算
        self.eps_chunk = self.epsilon * 0.1
        self.eps_seq = self.epsilon * 0.9
        
        self.laplace_mech = StandardLaplaceMechanism(
            epsilon=self.eps_seq, 
            sensitivity=self.sensitivity
        )
        self.scale = self.laplace_mech.scale
        
        safe_delta = min(self.delta, 0.5) 
        self.T = (2 * self.scale) * np.log(self.D / (2 * safe_delta))
        
        print(f"[Init] Streaming Parallel Partition (Ultimate Optimization)")
        print(f"[Init] Workers: {self.num_threads}, Target T: {self.T:.4f}")

    def get_seal_probability(self, current_load, z_init):
        distance = self.T + z_init - current_load
        if distance >= 0:
            return min(1.0, max(0.0, 0.5 * np.exp(-distance / self.scale)))
        else:
            return min(1.0, max(0.0, 1.0 - 0.5 * np.exp(distance / self.scale)))

    def _optimize_worker_count(self, N):
        """自适应并发数安全约束"""
        K = self.num_threads
        if K <= 1: return 1

        original_K = K
        # 1. 阈值饥饿约束
        if (N / K) < 3 * self.T:
            K = max(1, int(N / (3 * self.T)))
        print(f"阈值饥饿约束: {int(N / (3 * self.T))}")
        # 2. 块噪声吞噬约束
        noise_scale = self.sensitivity / self.eps_chunk
        if (N / K) < 10 * noise_scale:
            K = min(K, max(1, int(N / (10 * noise_scale))))
        print(f"块噪声吞噬约束: {int(N / (10 * noise_scale))}")
        # 3. IPC 系统开销约束
        if N < 50_000:
            K = 1

        # if K < original_K:
        print(f"  [系统提示] 触发安全约束，并发数自适应降级: {original_K} -> {K}")
        return K

    def _local_partition(self, raw_chunk):
        """
        [Worker 核心：流式扫描] 
        单遍 O(N) 扫描，无字典，无排序。
        前提：传入的 raw_chunk 必须是递增有序的数组。
        """
        output_splits = []
        if not len(raw_chunk):
            return output_splits
            
        current_load = 0
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        # 冷启动：避免产生头部的虚假跳跃，假装是从上一个紧挨着的位置过来的
        last_processed_key = raw_chunk[0] - 1 
        active_key = raw_chunk[0]
        active_freq = 0
        
        def process_key_event(key, freq, last_pos):
            """处理 Key 边界触发的 SVT 事件"""
            nonlocal current_load, z_init, target
            
            # --- Part 1: 局部 Gap 处理 ---
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
                    except: pass
            
            # --- Part 2: 真实 Key 处理 ---
            current_load += freq
            z_check = self.laplace_mech.generate_noise()
            if current_load + z_check > target:
                output_splits.append(int(key))
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            return key # 刷新位置

        # --- 核心：极速 O(N) 流式遍历 ---
        for k in raw_chunk:
            if k == active_key:
                active_freq += 1
            else:
                # 遇到不同的 Key，触发清算
                last_processed_key = process_key_event(active_key, active_freq, last_processed_key)
                # 记录新 Key
                active_key = k
                active_freq = 1
                
        # 循环结束，清算最后一个积压的 Key
        last_processed_key = process_key_event(active_key, active_freq, last_processed_key)

        # 核心：局部停机！绝对不向 D 模拟尾部 Gap！
        return output_splits

    def run_partition(self, raw_keys):
        """
        [Master 节点]
        接收全局有序的 raw_keys，执行带噪切片并分发。
        """
        N = len(raw_keys)
        if N == 0:
            return self._format_output([self.D])
            
        K = self._optimize_worker_count(N)
        
        if K <= 1:
            splits = self._local_partition(raw_keys)
            splits.append(self.D)
            return self._format_output(sorted(list(set(splits))))

        # =================================================
        # Stage 1: DP Data Chunking (只切分物理 Index)
        # =================================================
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

        # =================================================
        # Stage 2: 并行流式处理 (Data Parallelism)
        # =================================================
        global_splits = []
        futures = []
        
        with ProcessPoolExecutor(max_workers=K) as executor:
            for i in range(K):
                start_idx = indices[i]
                end_idx = indices[i+1]
                if start_idx == end_idx:
                    continue 
                    
                # 零拷贝级别的分片：仅仅是底层数组视图或极轻量级的切片
                chunk_raw_data = raw_keys[start_idx:end_idx]
                futures.append(executor.submit(self._local_partition, chunk_raw_data))

        # =================================================
        # Stage 3: Master 无脑缝合去重
        # =================================================
        for future in futures:
            global_splits.extend(future.result())
            
        # 物理兜底并消除同 Key 被切碎产生的边缘重叠
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