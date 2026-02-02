import numpy as np
from collections import Counter

class PrivatePartition:
    def __init__(self, epsilon, delta, domain_size):
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        
        # 1. 计算噪声安全边界 M
        # M = (1/eps) * ln(1/delta)
        self.noise_bound = (1.0 / self.epsilon) * np.log(1.0 / self.delta)
        
        # 2. 计算阈值 T
        # T >= (2/eps) * ln(D/2delta)
        # 保证 T > 2M，从而保证 distance > M (在 Gap 处)
        self.T = (2 / self.epsilon) * np.log(self.D / (2 * self.delta))
        
        print(f"[Init] Params: T={self.T:.2f}, NoiseBound(M)={self.noise_bound:.2f}")

    def get_bounded_laplace_noise(self):
        """生成在 [-M, M] 截断的 Laplace 噪音"""
        raw_noise = np.random.laplace(0, 1.0 / self.epsilon)
        return np.clip(raw_noise, -self.noise_bound, self.noise_bound)

    def get_geometric_prob(self, distance):
        """计算概率，用于 -M <= distance <= M 的情况"""
        scale = 1.0 / self.epsilon
        if distance >= 0:
            p = 0.5 * np.exp(-distance / scale)
        else:
            p = 1.0 - 0.5 * np.exp(distance / scale)
        return np.clip(p, 1e-20, 1.0 - 1e-10)

    def preprocess_data(self, raw_keys):
        if not raw_keys: return []
        counter = Counter(raw_keys)
        return sorted(counter.items(), key=lambda x: x[0])

    def run_partition(self, sorted_data):
        output_splits = []
        current_load = 0
        
        # 初始化带噪阈值 (仅限制噪音部分)
        threshold_noise = self.get_bounded_laplace_noise()
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
            
            # 显式截断查询噪音
            query_noise = self.get_bounded_laplace_noise()
            
            if current_load + query_noise > noisy_threshold:
                output_splits.append(key)
                current_load = 0
                threshold_noise = self.get_bounded_laplace_noise()
                noisy_threshold = self.T + threshold_noise
            
            # -------------------------------------------------
            # Part B: Gap Part (空隙)
            # -------------------------------------------------
            gap_len = next_key - key - 1
            
            if gap_len > 0:
                distance = noisy_threshold - current_load
                
                # 【一致性检查】：模拟 Query Noise 的截断效果
                if distance > self.noise_bound:
                    # 距离 > M，最大噪音 M 也无法触发切分
                    pass # 相当于 p=0
                
                elif distance < -self.noise_bound:
                    # 距离 < -M，最小噪音 -M 也能触发切分
                    # 必定切分
                    cut_location = key + 1
                    output_splits.append(cut_location)
                    
                    current_load = 0
                    threshold_noise = self.get_bounded_laplace_noise()
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
                        threshold_noise = self.get_bounded_laplace_noise()
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