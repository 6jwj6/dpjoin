import numpy as np
import pandas as pd
from collections import Counter

class PrivatePartition:
    def __init__(self, epsilon, delta, domain_size):
        """
        初始化隐私参数
        :param epsilon: 隐私预算
        :param delta: 失败概率 (用于计算 Gap 的安全阈值)
        :param domain_size: 全域大小 D (Key 的取值范围 [1, D])
        """
        self.epsilon = epsilon
        self.delta = delta
        self.D = domain_size
        
        # 根据推导公式计算阈值 T
        # T >= (2/epsilon) * ln(D / (2*delta))
        self.T = (2 / self.epsilon) * np.log(self.D / (2 * self.delta))
        
        # 记录用于 debug 或分析的理论噪声界限 M (Key Part)
        # 假设 beta = delta (为了简化)
        # M = (1/epsilon) * ln(2N/beta) 这个在运行时无法预知N，通常用于事后分析accuracy
        
    def add_laplace_noise(self, scale):
        """生成拉普拉斯噪声 Lap(scale)"""
        return np.random.laplace(0, scale)

    def get_geometric_prob(self, distance):
        """
        计算单点切分概率 p = Pr[Lap(1/eps) > distance]
        distance = Threshold - Current_Load
        """
        # 注意：这里使用的是 b = 1/epsilon 的拉普拉斯分布
        scale = 1.0 / self.epsilon
        
        if distance >= 0:
            return 0.5 * np.exp(-distance / scale)
        else:
            return 1.0 - 0.5 * np.exp(distance / scale)

    def preprocess_data(self, raw_keys):
        """
        Step 1: 扫描数据库，统计 (key, freq) 并排序
        :param raw_keys: 数据库中的原始 Key 列表 (如 [1, 5, 1, 8, ...])
        :return: 排序后的列表 [(key1, f1), (key2, f2), ...]
        """
        print("正在进行数据预处理...")
        counter = Counter(raw_keys)
        # 按 Key 从小到大排序
        sorted_data = sorted(counter.items(), key=lambda x: x[0])
        print(f"统计完成。非零 Key 数量 N = {len(sorted_data)}")
        return sorted_data

    def run_partition(self, sorted_data):
        """
        Step 2: 核心 Partition 算法 (Gap-Truncated SVT)
        """
        output_splits = []
        
        # 初始化状态
        current_load = 0
        # 初始阈值带噪声
        threshold_noise = self.add_laplace_noise(1.0 / self.epsilon)
        noisy_threshold = self.T + threshold_noise
        
        # 为了处理方便，我们在数据末尾追加一个虚拟的边界点 D+1
        # 这样最后一个真实数据之后的 Gap 也能被统一处理
        processing_list = sorted_data + [(self.D + 1, 0)]
        
        print(f"开始 Partition，基础阈值 T = {self.T:.4f} (带噪初始值: {noisy_threshold:.4f})")
        
        # 遍历每一个真实存在的 Key
        for i in range(len(processing_list) - 1):
            key, freq = processing_list[i]
            next_key = processing_list[i+1][0]
            
            # -------------------------------------------------
            # Part A: 处理真实数据 (Key Part)
            # -------------------------------------------------
            current_load += freq
            query_noise = self.add_laplace_noise(1.0 / self.epsilon)
            
            # 检查是否切分
            if current_load + query_noise > noisy_threshold:
                # 触发切分
                print(f"  [Key Cut] 在 Key={key} 处切分 (Load={current_load}, Noise={query_noise:.2f})")
                output_splits.append(key)
                
                # 重置状态
                current_load = 0
                threshold_noise = self.add_laplace_noise(1.0 / self.epsilon)
                noisy_threshold = self.T + threshold_noise
            
            # -------------------------------------------------
            # Part B: 处理空隙 (Gap Part)
            # -------------------------------------------------
            gap_len = next_key - key - 1
            
            if gap_len > 0:
                # 1. 基于当前 Load 计算切分概率
                # distance = 距离阈值还差多少
                distance = noisy_threshold - current_load
                p = self.get_geometric_prob(distance)
                
                # 2. 几何采样：模拟 "第几次 check 会成功"
                # np.random.geometric(p) 返回的是第一次成功的试验次数 (1, 2, ...)
                steps_to_cut = np.random.geometric(p)
                
                # 3. 判断切分点是否落在 Gap 内
                if steps_to_cut <= gap_len:
                    # 触发切分 (Gap Cut)
                    cut_location = key + steps_to_cut
                    print(f"  [Gap Cut] 在 Gap={key+1}~{next_key-1} 内切分，位置 {cut_location}")
                    output_splits.append(cut_location)
                    
                    # 重置状态
                    current_load = 0
                    threshold_noise = self.add_laplace_noise(1.0 / self.epsilon)
                    noisy_threshold = self.T + threshold_noise
                    
                    # 【关键点】：Truncate
                    # 既然在这个 Gap 里切了一刀，按照算法逻辑，我们直接放弃 Gap 剩余部分的检测
                    # 直接跳到 next_key 继续处理
                else:
                    # Gap 内没有发生切分，Load 保持不变，带入下一个 Key
                    pass

        # 整理输出区间
        # 输出形式通常是区间列表: [start, end]
        final_partitions = []
        start = 1
        for split_point in output_splits:
            # 确保不越界
            end = min(split_point, self.D)
            if start <= end:
                final_partitions.append((start, end))
            start = end + 1
            
        # 加上最后一段
        if start <= self.D:
            final_partitions.append((start, self.D))
            
        return final_partitions

# ==========================================
# 模拟运行
# ==========================================

if __name__ == "__main__":
    # 1. 模拟数据库表输入 (Raw Data)
    # 假设全域 D=100
    # 数据集中在某些区域，有大的 Gap
    raw_db_table = []
    # 区域 1: 1-10 比较密集
    for k in range(1, 11):
        raw_db_table.extend([k] * np.random.randint(1, 5))
    # 区域 2: 50-55 (中间有个大 Gap 11-49)
    for k in range(50, 56):
        raw_db_table.extend([k] * np.random.randint(5, 10)) # 这里频率高一点
    # 区域 3: 90-100
    for k in range(90, 101):
        raw_db_table.extend([k] * np.random.randint(1, 3))

    # 2. 设置参数
    D = 100
    epsilon = 1.0
    delta = 1e-5  # 通常设为 1/D^c

    # 3. 实例化算法
    algo = PrivatePartition(epsilon, delta, D)

    # 4. 执行第一步：统计
    sorted_stats = algo.preprocess_data(raw_db_table)
    # 打印部分统计结果看看
    print(f"前5个统计数据: {sorted_stats[:5]} ...")

    # 5. 执行第二步：Partition
    partitions = algo.run_partition(sorted_stats)

    print("\n最终划分结果 (Partitions):")
    for p in partitions:
        print(f"区间: {p}, 长度: {p[1]-p[0]+1}")


    