import numpy as np
from collections import Counter
from noise_mechanisms import BoundedLaplaceMechanism

class PrivatePartition:
    def __init__(self, epsilon, delta, domain_size):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        
        # 1. 实例化噪声机制 (委托给外部类处理)
        # 这会自动计算内部的 noise_bound
        self.laplace_mech = BoundedLaplaceMechanism(epsilon, delta)
        
        # 为了代码引用的方便，可以做一个别名，或者直接用 self.laplace_mech.noise_bound
        self.noise_bound = self.laplace_mech.noise_bound
        
        # 2. 计算阈值 T (这是 Partition 算法特有的逻辑，保留在这里)
        # T >= (2/eps) * ln(D/2delta)
        self.T = (2 / self.epsilon) * np.log(self.D / (2 * self.delta))
        
        print(f"[Init] Params: T={self.T:.2f}, NoiseBound(M)={self.noise_bound:.2f}")

    def get_geometric_prob(self, distance):
        """
        计算单点切分概率 p = Pr[Lap(1/eps) > distance]
        保留在 Partition 类中，因为它描述的是 gap 的切分行为。
        """
        scale = 1.0 / self.epsilon
        if distance >= 0:
            p = 0.5 * np.exp(-distance / scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / scale)
        
        # 保护防止 p 越界
        return np.clip(p, 1e-20, 1.0 - 1e-10)

    def preprocess_data(self, raw_keys):
        """预处理数据：统计并排序"""
        if len(raw_keys) == 0:
            return []
        counter = Counter(raw_keys)
        return sorted(counter.items(), key=lambda x: x[0])

    def run_partition(self, sorted_data):
        output_splits = []
        current_load = 0
        
        # --- 初始化阈值 ---
        # 1. 使用外部机制生成截断噪声
        threshold_noise = self.laplace_mech.generate_noise()
        noisy_threshold = self.T + threshold_noise
        
        # 添加虚拟边界处理尾部 Gap
        processing_list = sorted_data + [(self.D + 1, 0)]
        
        for i in range(len(processing_list) - 1):
            key, freq = processing_list[i]
            next_key = processing_list[i+1][0]
            
            # -------------------------------------------------
            # Part A: Key Part (真实数据)
            # -------------------------------------------------
            current_load += freq
            
            # 使用外部机制生成截断查询噪声
            query_noise = self.laplace_mech.generate_noise()
            
            if current_load + query_noise > noisy_threshold:
                output_splits.append(key)
                
                # 重置状态
                current_load = 0
                threshold_noise = self.laplace_mech.generate_noise()
                noisy_threshold = self.T + threshold_noise
            
            # -------------------------------------------------
            # Part B: Gap Part (空隙)
            # -------------------------------------------------
            gap_len = next_key - key - 1
            
            if gap_len > 0:
                distance = noisy_threshold - current_load
                
                # 【一致性检查】：使用 mechanism 中定义的边界
                if distance > self.noise_bound:
                    # 距离太远，最大噪音也无法触发
                    pass 
                
                elif distance < -self.noise_bound:
                    # 距离极小，必定触发
                    cut_location = key + 1
                    output_splits.append(cut_location)
                    
                    current_load = 0
                    threshold_noise = self.laplace_mech.generate_noise()
                    noisy_threshold = self.T + threshold_noise
                    # Truncate Gap -> 跳过剩余
                    
                else:
                    # 距离在可达范围内，采样几何分布
                    p = self.get_geometric_prob(distance)
                    steps_to_cut = np.random.geometric(p)
                    
                    if steps_to_cut <= gap_len:
                        cut_location = key + steps_to_cut
                        output_splits.append(cut_location)
                        current_load = 0
                        threshold_noise = self.laplace_mech.generate_noise()
                        noisy_threshold = self.T + threshold_noise
                    else:
                        pass # Gap 内未发生切分

        # 整理输出
        final_partitions = []
        start = 1
        for split_point in output_splits:
            end = min(split_point, self.D)
            if start <= end:
                final_partitions.append((start, end))
            start = end + 1
            
        if start <= self.D:
            final_partitions.append((start, self.D))
            
        return final_partitions