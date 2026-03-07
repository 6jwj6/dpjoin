import numpy as np
from collections import Counter
from noise_mechanisms import BoundedLaplaceMechanism

class PrivatePartition:
    def __init__(self, epsilon, delta, domain_size, sensitivity=1):
        """
        初始化 Partition 算法 (Truncated Offline Version)
        
        Args:
            epsilon: 隐私预算 epsilon
            delta: 隐私预算 delta
            domain_size: 域大小 D (虽然不需要均摊，但用于处理边界 gap)
            sensitivity: 敏感度 (默认为 1)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        self.sensitivity = sensitivity
        
        # -------------------------------------------------------
        # 【核心修改】并行组合 (Parallel Composition)
        # -------------------------------------------------------
        # 之前的逻辑：self.delta_local = self.delta / (2 * self.D)
        # 现在的逻辑：由于不同分段的数据变动互不影响，隐私损失是局部的。
        # 我们只需要保证单次判定（涉及 2 个噪声实例）的失效概率不超过 delta。
        # 因此，只需要将 delta 平分给 "阈值噪声" 和 "计数噪声" 
        # self.delta_local = self.delta / 2.0
        # 按照原论文证明，甚至不需要平分delta给 阈值噪声 和 计数噪声
        # 目前这样单边是 0.5delta
        self.delta_local = self.delta
        # 实例化拉普拉斯机制
        # 注意：BoundedLaplaceMechanism 内部应使用 A = (Delta * ln(1/delta_local)) / epsilon 计算边界
        self.laplace_mech = BoundedLaplaceMechanism(
            epsilon=self.epsilon, 
            delta=self.delta_local, 
            sensitivity=self.sensitivity
        )
        
        # 同步边界与阈值
        self.noise_bound = self.laplace_mech.noise_bound
        
        # 阈值 T 仍然保持 3倍边界，保证 count=0 时概率严格为 0 (Gap跳跃的前提)
        self.T = 3 * self.noise_bound
        
        print(f"[Init] Sensitivity={self.sensitivity}, Global Delta={self.delta}")
        print(f"[Init] Local Delta={self.delta_local:.2e} (No division by D!)")
        print(f"[Init] Noise Bound (M)={self.noise_bound:.4f}, Threshold (T)={self.T:.4f}")

    def get_geometric_prob(self, count, z_init):
        """
        计算在当前负载和阈值噪声下，下一个位置密封的概率 p。
        基于截断拉普拉斯的几何近似。
        """
        distance = self.T + z_init - count
        
        # 【防御 1】物理硬截断：如果距离大于 M，说明需要跨越的距离超过了噪声最大可能值
        # 在截断机制下，这意味着密封概率绝对为 0。
        # 这也是 Offline 算法能 "跳过 Gap" 的数学基础。
        if distance >= self.noise_bound:
            return 0.0
            
        # 如果距离极小（已经由负值超过边界），概率为 1，必须密封
        if distance <= -self.noise_bound:
            return 1.0
            
        scale = self.sensitivity / self.epsilon
        
        # 标准拉普拉斯累积分布函数 (CDF) 的变体，用于计算超过阈值的概率
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
        
        # 初始化第一个段的阈值噪声
        z_init = self.laplace_mech.generate_noise()
        target = self.T + z_init
        
        last_pos = 0 
        
        for key, freq in sorted_data:
            # -------------------------------------------------
            # 步骤 1: 处理当前 Key 之前的 Gap (全 0 区域)
            # -------------------------------------------------
            gap_len = key - last_pos - 1
            if gap_len > 0:
                # 在 Gap 中，count 为 0（相对于当前段累积的 current_load）
                # 注意：这里的 current_load 是指从上一个切点到现在积累的非零值
                # 在 Gap 内部，累积值保持不变
                p = self.get_geometric_prob(current_load, z_init)
                
                # 【防御 2】只有 p > 0 时才尝试采样几何分布
                # 由于我们有物理硬截断 (distance >= bound 返回 0)，
                # 这里的 p 经常会是严格的 0.0，直接跳过整个 Gap，效率极高。
                if p > 0: 
                    try:
                        # 几何分布模拟：在概率 p 下，多少次试验能成功一次
                        steps = np.random.geometric(p)
                        
                        # 如果步数落在 Gap 内，说明在 Gap 中间发生了切分
                        if steps <= gap_len:
                            cut_pos = last_pos + steps
                            output_splits.append(int(cut_pos))
                            
                            # 重置状态
                            current_load = 0
                            z_init = self.laplace_mech.generate_noise()
                            target = self.T + z_init
                            
                            # 注意：理论上 Gap 剩余部分可能再次切分。
                            # 但由于重置后 current_load=0 且 T > 2A，
                            # 新的 p 将变为 0，因此不需要递归处理剩余 Gap。
                            # 这是一个巨大的性能优化。
                    except ValueError:
                        pass # 防止 p 极小时 geometric 抛出异常
            
            # -------------------------------------------------
            # 步骤 2: 处理 Key 节点 (非零数据)
            # -------------------------------------------------
            current_load += freq
            
            # 生成计数噪声 (Check Noise)
            z_check = self.laplace_mech.generate_noise()
            
            # 判定是否密封
            # 逻辑等价于：freq_sum + noise_check > T + noise_init
            if current_load + z_check > target:
                output_splits.append(int(key))
                current_load = 0
                z_init = self.laplace_mech.generate_noise()
                target = self.T + z_init
            
            last_pos = key

        # -------------------------------------------------
        # 步骤 3: 处理最后一个 Key 到 D 之间的 Final Gap
        # -------------------------------------------------
        final_gap = self.D - last_pos
        if final_gap > 0:
            p = self.get_geometric_prob(current_load, z_init)
            if p > 0:
                try:
                    steps = np.random.geometric(p)
                    if steps <= final_gap:
                        output_splits.append(int(last_pos + steps))
                except:
                    pass

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