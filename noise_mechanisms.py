import numpy as np
import math

class ShiftedTruncatedGeometricMechanism:
    """
    实现了 Definition 3.6: Shifted and Truncated Geometric Distribution.
    
    原理:
    该机制专门用于处理整数输出（如基数填充）。
    由于其输出范围必须是非负的 [0, Upper]，因此保留了 Shift 逻辑。
    """
    def __init__(self, epsilon: float, delta: float, sensitivity: int = 1):
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
            
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # 1. 计算 k0 (安全尾部距离)
        # 满足 P[|Geom| >= k0] <= delta
        scale = self.sensitivity / self.epsilon
        self.k0 = math.ceil(scale * math.log(2.0 / self.delta))
        
        # 2. 计算偏移中心 (Shift) 和 上界 (Upper Bound)
        # 在离散整数空间中，为了覆盖 Delta 个“消失点”，偏移量为 k0 + Delta - 1
        self.shift = self.k0 + self.sensitivity - 1
        
        # 对应图中的支撑集范围 [0, 2(k0 + Delta - 1)]
        self.upper_bound = 2 * self.shift
        
        # 几何分布概率参数
        self.geom_p = 1.0 - np.exp(-self.epsilon / self.sensitivity)

    def sample_two_sided_geometric(self) -> int:
        """采样双边几何分布 (Discrete Laplace)"""
        g1 = np.random.geometric(self.geom_p)
        g2 = np.random.geometric(self.geom_p)
        return int(g1 - g2)

    def generate_noise(self) -> int:
        """生成最终噪声值 (非负整数，带偏置)"""
        raw_geom = self.sample_two_sided_geometric()
        shifted_val = self.shift + raw_geom
        # 边界重映射 (Clamping)
        return int(np.clip(shifted_val, 0, self.upper_bound))
    
    def get_max_noise(self) -> int:
        return self.upper_bound


class BoundedLaplaceMechanism:
    """
    实现了对称截断拉普拉斯分布 (Symmetric Truncated Laplace).
    
    原理:
    不进行平移，均值为 0。
    通过在边界 M = k0 + sensitivity 处进行截断（钳制），
    确保在敏感度为 Delta 的变换下，落在范围外的概率质量累计不超过 delta。
    """
    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        self.scale = self.sensitivity / self.epsilon
        
        # 1. 计算基础安全边界 k0
        # 使得 P[|X| >= k0] <= delta，即单边尾部为 delta/2
        # 按照原论文的证明，其实只需要单边尾部是 delta 即可，可以再紧一点
        self.k0 = self.scale * math.log(1.0 / self.delta)
        
        # 2. 边界修正：在对称截断版本中，边界需要额外考虑 +sensitivity
        # 这是为了确保当真实值移动 Delta 时，重叠区域之外的概率和仍受 delta 保护
        self.noise_bound = self.k0 + self.sensitivity

    def generate_noise(self) -> float:
        """
        生成在 [-M, M] 之间对称截断的 Laplace 噪声。
        无偏置，均值为 0。
        """
        # 1. 采样原始连续噪声
        raw_noise = np.random.laplace(0, self.scale)
        
        # 2. 物理截断 (Clamping)
        # 将超出 [-noise_bound, noise_bound] 的值映射到边界上
        return float(np.clip(raw_noise, -self.noise_bound, self.noise_bound))

    def get_noise_bound(self) -> float:
        """返回噪声的物理截断上界 M"""
        return self.noise_bound
    

class StandardLaplaceMechanism:
    """
    【新增】实现了标准的拉普拉斯机制 (Standard Laplace Mechanism).
    
    原理:
    标准的 pure epsilon-DP 机制。噪声范围为 (-inf, +inf)。
    这里不进行物理截断，但在 Utility 分析中，我们使用 get_tail_bound 
    来计算噪声大概率(1-beta)不会超过的边界，用于设置阈值 T。
    """
    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        # 注意：标准 Laplace 不需要 delta 参数来定义分布，只依赖 epsilon
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = self.sensitivity / self.epsilon
        
    def generate_noise(self) -> float:
        """生成标准拉普拉斯噪声 (无截断)"""
        return np.random.laplace(0, self.scale)
    
    def get_tail_bound(self, confidence_prob: float) -> float:
        """
        辅助函数：计算概率尾部边界 (Utility Bound).
        
        返回一个边界 M，使得 Pr[|Noise| > M] <= confidence_prob.
        这对应于 Online 算法推导中，为了保证 1-beta 的 Utility 所需计算的阈值分量。
        
        公式: Pr[|X| > M] = exp(-M/b) <= beta
             => M >= b * ln(1/beta)
        """
        if confidence_prob <= 0 or confidence_prob >= 1:
             raise ValueError("Confidence probability (beta/delta) must be in (0, 1)")
             
        return self.scale * math.log(1.0 / confidence_prob)