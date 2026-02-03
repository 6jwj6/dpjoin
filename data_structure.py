import numpy as np
from dataclasses import dataclass, field
from noise_mechanisms import ShiftedTruncatedGeometricMechanism

@dataclass
class Table:
    """
    代表一个数据库表，包含差分隐私所需的元数据。
    """
    name: str
    keys: np.ndarray          # 存储 Key (整数)
    payloads: np.ndarray      # 存储 Payload (暂时为空)
    
    # --- 隐私元数据 ---
    # 【修改点】敏感度严格为 int
    sensitivity: int = 1      
    
    # 真实的统计值
    true_max_freq: int = field(init=False) 
    
    # 加噪后的统计值 (用于 Join 敏感度计算)
    # 这是一个偏大的估计值 (True + ShiftedNoise)
    noisy_max_freq: int = field(init=False)

    def __post_init__(self):
        """初始化后自动计算真实的 Max Frequency"""
        if len(self.keys) > 0:
            # 使用 numpy 高效计算最大频率
            _, counts = np.unique(self.keys, return_counts=True)
            self.true_max_freq = int(counts.max())
        else:
            self.true_max_freq = 0
            
        # 初始状态下，noisy_max_freq 暂定为真实值
        # 实际使用前必须调用 privatize_metadata
        self.noisy_max_freq = self.true_max_freq

    def privatize_metadata(self, epsilon: float, delta: float):
        """
        使用 Shifted and Truncated Geometric Noise 更新 noisy_max_freq。
        
        :param epsilon: 用于元数据隐私的预算
        :param delta: 失败概率
        """
        # 1. 实例化噪声机制
        # Max Freq 的敏感度通常为 1 (增加一行，MaxFreq 最多 +1)
        # 但如果是中间表，sensitivity 可能已经被之前的计算放大了，所以这里传入 self.sensitivity
        mech = ShiftedTruncatedGeometricMechanism(epsilon, delta, sensitivity=self.sensitivity)
        
        # 2. 生成噪声
        # 注意：这里的噪声是一个正数 (包含 Shift 偏移)
        # 用它加到真实频率上，会得到一个“安全上界” (Safe Upper Bound)
        noise = mech.generate_noise()
        
        # 3. 更新 noisy_max_freq
        self.noisy_max_freq = self.true_max_freq + noise
        
        print(f"[{self.name}] Metadata Privatized: "
              f"True={self.true_max_freq}, Noise={noise}, Noisy={self.noisy_max_freq}")
        
        return self.noisy_max_freq

    def __repr__(self):
        return (f"Table(name='{self.name}', rows={len(self.keys)}, "
                f"μ={self.sensitivity}, "
                f"TrueFreq={self.true_max_freq}, NoisyFreq={self.noisy_max_freq})")

# ==========================================
# 简单的测试代码
# ==========================================
if __name__ == "__main__":
    # 生成一点模拟数据
    keys = np.array([1, 1, 1, 2, 2, 3, 4, 5, 5])
    payloads = np.full(len(keys), None)
    
    # 实例化 Table，假设这是某个 Join 后的中间结果，敏感度为 2
    t = Table("TestTable", keys, payloads, sensitivity=2)
    
    print("初始化状态:", t)
    
    # 加噪
    # epsilon=0.1, delta=1e-5
    t.privatize_metadata(0.1, 1e-5)
    
    print("加噪后状态:", t)
