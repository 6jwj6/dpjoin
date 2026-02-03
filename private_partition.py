import numpy as np
from collections import Counter
from noise_mechanisms import BoundedLaplaceMechanism

class PrivatePartition:
    def __init__(self, epsilon, delta, domain_size, sensitivity=1):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.sensitivity = sensitivity
        
        # 全域安全 delta 摊薄
        self.delta_local = self.delta / self.D
        
        # 实例化拉普拉斯机制
        self.laplace_mech = BoundedLaplaceMechanism(
            epsilon=self.epsilon, 
            delta=self.delta_local, 
            sensitivity=self.sensitivity
        )
        
        # 同步边界与阈值
        self.noise_bound = self.laplace_mech.noise_bound
        self.T = 3 * self.noise_bound
        
        print(f"[Init] M={self.noise_bound:.2f}, T={self.T:.2f}, D={self.D:,}")

    def get_geometric_prob(self, count, z_init):
        # distance = T + z_init - count
        distance = self.T + z_init - count
        
        # 【防御 1】物理硬截断：如果距离大于 M，概率理论上绝对为 0
        # 这一步防止了极小 p 导致的几何采样溢出
        if distance >= self.noise_bound:
            return 0.0
            
        # 如果距离极小，概率为 1
        if distance <= -self.noise_bound:
            return 1.0
            
        scale = self.sensitivity / self.epsilon
        if distance >= 0:
            p = 0.5 * np.exp(-distance / scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / scale)
            
        return np.clip(p, 0.0, 1.0)

    def preprocess_data(self, raw_keys):
        if len(raw_keys) == 0: return []
        counter = Counter(raw_keys)
        return sorted(counter.items(), key=lambda x: x[0])

    def run_partition(self, sorted_data):
        output_splits = []
        current_load = 0
        
        # 初始化
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        # 记录上一个处理过的位置，用于计算 Gap
        last_pos = 0 
        
        for key, freq in sorted_data:
            # -------------------------------------------------
            # 步骤 1: 处理当前 Key 之前的 Gap
            # -------------------------------------------------
            gap_len = key - last_pos - 1
            if gap_len > 0:
                p = self.get_geometric_prob(current_load, z_init)
                # 【防御 2】只有 p 足够大时才采样
                if p > 1e-15: 
                    try:
                        steps = np.random.geometric(p)
                        if steps <= gap_len:
                            cut_pos = last_pos + steps
                            output_splits.append(int(cut_pos))
                            current_load = 0
                            z_init = self.laplace_mech.generate_noise()
                            target = self.T + z_init
                            # 递归处理 Gap 剩余部分（简化处理：Gap 内最多切一次）
                    except:
                        pass # 溢出保护
            
            # -------------------------------------------------
            # 步骤 2: 处理 Key 节点
            # -------------------------------------------------
            current_load += freq
            z_check = self.laplace_mech.generate_noise()
            
            if current_load + z_check > target:
                output_splits.append(int(key))
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            last_pos = key

        # 处理最后一个 Key 到 D 之间的 Gap
        final_gap = self.D - last_pos
        if final_gap > 0:
            p = self.get_geometric_prob(current_load, z_init)
            if p > 1e-15:
                steps = np.random.geometric(p)
                if steps <= final_gap:
                    output_splits.append(int(last_pos + steps))

        return self._format_output(output_splits)

    def _format_output(self, output_splits):
        # 确保有序且在域内
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