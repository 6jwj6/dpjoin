import numpy as np
import math

class ShiftedTruncatedGeometricMechanism:
    """
    实现了 Definition 3.6: Shifted and Truncated Geometric Distribution.
    主要用于:
    1. Table 的 Max Frequency 加噪 (整数)
    2. Bucket 的 Padding Size 计算 (整数)
    3. Gap 的 Cut 模拟 (整数)
    """
    def __init__(self, epsilon: float, delta: float, sensitivity: int = 1):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # 1. 计算 k0
        # k0 是最小的正整数，使得 Pr[|Geom| >= k0] <= delta
        # 公式: k0 ≈ (Delta / epsilon) * ln(2 / delta)
        # 这里的 Geom 是双边几何分布
        term = (self.sensitivity / self.epsilon) * math.log(2.0 / self.delta)
        self.k0 = math.ceil(term)
        
        # 2. 计算偏移中心 (Shift) 和 上界 (Upper Bound)
        # Shift = k0 + Delta - 1
        self.shift = self.k0 + self.sensitivity - 1
        self.upper_bound = 2 * self.shift
        
        # 3. 计算几何分布参数
        # alpha = exp(-epsilon / Delta)
        # numpy.random.geometric(p) 需要成功概率 p = 1 - alpha
        self.alpha = np.exp(-self.epsilon / self.sensitivity)
        self.geom_p = 1.0 - self.alpha

    def sample_two_sided_geometric(self) -> int:
        """采样双边几何分布 Y = G1 - G2"""
        g1 = np.random.geometric(self.geom_p)
        g2 = np.random.geometric(self.geom_p)
        return int(g1 - g2)

    def generate_noise(self) -> int:
        """
        生成最终的噪声值 (非负整数)
        Result = min( max(0, Shift + TwoSidedGeom), UpperBound )
        注意：这个值本身包含了一个较大的正均值 (Shift)，
        用于 Upper Bound 估计时是安全的(高估)。
        """
        raw_geom = self.sample_two_sided_geometric()
        shifted_val = self.shift + raw_geom
        
        # 截断逻辑
        val_non_negative = max(0, shifted_val)
        final_val = min(val_non_negative, self.upper_bound)
        
        return int(final_val)


class BoundedLaplaceMechanism:
    """
    实现了截断拉普拉斯分布。
    主要用于:
    1. Partition 阶段的 Threshold 加噪 (浮点数)
    2. Partition 阶段的 Query 加噪 (浮点数)
    """
    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # 计算安全边界 M (Noise Bound)
        # Pr[|Lap| > M] <= delta
        # M = (Delta / epsilon) * ln(1 / delta)
        self.noise_bound = (self.sensitivity / self.epsilon) * np.log(1.0 / self.delta)

    def generate_noise(self) -> float:
        """生成在 [-M, M] 之间截断的 Laplace 噪声"""
        raw_noise = np.random.laplace(0, self.sensitivity / self.epsilon)
        return float(np.clip(raw_noise, -self.noise_bound, self.noise_bound))
