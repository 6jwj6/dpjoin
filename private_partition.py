import numpy as np
from collections import Counter
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
        # 依据公式: T >= (2 / epsilon) * ln(D / (2 * delta))
        # 
        # 推导回顾:
        # 我们需要 Pr[Any Noise Difference > T] <= delta
        # Noise Difference (Check - Init) 服从两个拉普拉斯之差的分布。
        # 系数 2 来自于 (1/epsilon) * 2，因为我们要压制的是两个噪声的波动。
        # 
        # 注意：这个 T 是为了 Privacy (Delta) 服务的，而不是 Utility (Beta)。
        # Utility 的 T 通常更小，但为了算法合法性，我们必须取这个较大的 T。
        
        safe_delta = min(self.delta, 0.5) # 防止数值错误
        self.T = (2 * self.scale) * np.log(self.D / (2 * safe_delta))
        
        print(f"[Init] Offline Partition (Standard Laplace)")
        print(f"[Init] Epsilon={self.epsilon}, Delta(Privacy)={self.delta}")
        print(f"[Init] Privacy Threshold (T)={self.T:.4f}")

    def get_seal_probability(self, current_load, z_init):
        """
        计算在当前状态下发生密封(Seal)的概率。
        
        判定不等式: current_load + z_check > T + z_init
        
        我们将 z_init 视为当前分段的常数（因为它在分段开始时已固定）。
        我们计算随机变量 z_check 超过 (T + z_init - current_load) 的概率。
        
        Args:
            current_load: 当前累积的真实计数
            z_init: 当前分段的初始阈值噪声
        """
        # 距离目标还有多远
        distance = self.T + z_init - current_load
        
        # 计算 Pr[z_check > distance]
        # 使用标准拉普拉斯 CDF 的尾部公式
        # Pr(X > d) = 0.5 * exp(-d/b)       if d >= 0
        # Pr(X > d) = 1 - 0.5 * exp(d/b)    if d < 0
        
        if distance >= 0:
            p = 0.5 * np.exp(-distance / self.scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / self.scale)
            
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
        # 此时的判定目标
        target = self.T + z_init
        
        last_pos = 0 
        
        for key, freq in sorted_data:
            # =================================================
            # Part 1: Gap 处理 (The Offline Optimization)
            # =================================================
            gap_len = key - last_pos - 1
            
            if gap_len > 0:
                # 在 Gap 中，Real Count 为 0
                # 计算单步切分概率 p
                p = self.get_seal_probability(current_load, z_init)
                
                # 【数值稳定性守卫】
                # 由于 T 是根据 Privacy Delta 计算的，通常很大 (e.g. 30+)。
                # 导致 p 极小 (e.g. 1e-20)。
                # 这种情况下，期望发生次数 ~ D * 1e-20 ≈ 0，可以直接跳过。
                if p > 1e-15:
                    try:
                        # 核心：采样几何分布，看哪一步会切分
                        steps = np.random.geometric(p)
                        
                        if steps <= gap_len:
                            # 发生了切分
                            cut_pos = last_pos + steps
                            output_splits.append(int(cut_pos))
                            
                            # 【重要逻辑】
                            # 根据我们之前的讨论：一个 Gap 最多只处理一次切分。
                            # 剩余的 Gap (全0) 再次切分的概率被包含在 Delta 风险中，
                            # 因此我们可以安全地忽略剩余部分，重置状态。
                            current_load = 0
                            z_init = self.laplace_mech.generate_noise()
                            target = self.T + z_init
                            
                            # Gap 处理结束，进入 Key 处理
                    except ValueError:
                        pass # p 极小时 geometric 可能抛出异常
            
            # =================================================
            # Part 2: Key 处理 (Standard Online Logic)
            # =================================================
            current_load += freq
            
            # 生成当前时刻的检查噪声
            z_check = self.laplace_mech.generate_noise()
            
            # 判定是否切分
            if current_load + z_check > target:
                output_splits.append(int(key))
                
                # 重置状态
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