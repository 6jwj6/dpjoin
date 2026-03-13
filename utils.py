import numpy as np
import time
from data_structure import Table
from private_partition import PrivatePartitionOffline as PrivatePartition 
from bucket_mechanism import BucketProcessor


# 1. 核心辅助函数
def preprocess_table_data(records, join_key_name):
    """
    将包含完整属性的字典列表，按指定的 Join Key 转化为 DP 算法需要的格式。
    输出: [(key, freq, list_of_payloads), ...]
    """
    from collections import defaultdict
    data_map = defaultdict(list)
    for row in records:
        k = row[join_key_name]
        data_map[k].append(row)
        
    sorted_data = [(k, len(p_list), p_list) for k, p_list in sorted(data_map.items(), key=lambda x: x[0])]
    return sorted_data

def extract_flat_table_from_buckets(buckets, dummy_key=-999):
    """
    从带有 DP 噪声的 Bucket 中提取出平铺的明文表。
    过滤掉 Dummy 数据，提取出所有真实的组合 Payload 字典。
    """
    flat_records = []
    for bucket in buckets:
        for k, f, p_list in bucket:
            if k != dummy_key:
                flat_records.extend(p_list)
    return flat_records


# 2. 打印与统计模块
def print_summary(name: str, partitions, final_buckets, elapsed_time, D: int):
    """
    打印单表详细统计报告
    """
    num_buckets = len(final_buckets)
    total_real_items = 0
    total_dummy_items = 0
    empty_buckets = 0
    
    for bucket in final_buckets:
        # 1. 统计真实 Key 的 Tuple 数量（用于倒推 Dummy 数量）
        real_tuple_count = sum(1 for k, f, p in bucket if k != -999)
        # 2. 统计真实的记录总条数（累加 frequency）
        real_items_in_bucket = sum(f for k, f, p in bucket if k != -999)
        
        # 列表总长度减去真实 Tuple 数量，就是 Dummy 的填充数量
        dummies_in_bucket = len(bucket) - real_tuple_count
        
        total_real_items += real_items_in_bucket
        total_dummy_items += dummies_in_bucket
        if real_items_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_items + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    
    print(f"\n" + "-"*60)
    print(f" [{name}] Partition & Bucket 处理报告")
    print(f"-"*60)
    print(f"  - 算法耗时:           {elapsed_time:.4f} 秒")
    print(f"  - 生成区间数:         {len(partitions):,}")
    print(f"  - 纯噪声桶 (Empty):   {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"  - 真实数据 (Real):    {total_real_items:,} 项")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,} 项")
    if total_real_items > 0:
        print(f"  - 膨胀系数 (Ratio):   {(total_real_items + total_dummy_items)/total_real_items:.2f}x")
    print(f"  - 平均 Bucket 大小:   {avg_bucket_size:.2f} items")


def print_join_summary(joined_partitions, joined_buckets, elapsed_time):
    """
    打印 Join 操作的统计报告
    """
    num_buckets = len(joined_buckets)
    total_real_items = 0
    total_dummy_items = 0
    empty_buckets = 0
    
    for bucket in joined_buckets:
        # 同理，拆分 Tuple 计数和 Freq 累加
        real_tuple_count = sum(1 for k, f, p in bucket if k != -999)
        real_items_in_bucket = sum(f for k, f, p in bucket if k != -999)
        
        dummies_in_bucket = len(bucket) - real_tuple_count
        
        total_real_items += real_items_in_bucket
        total_dummy_items += dummies_in_bucket
        if real_items_in_bucket == 0:
            empty_buckets += 1
            
    avg_bucket_size = (total_real_items + total_dummy_items) / num_buckets if num_buckets > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"      差分隐私 Join 阶段压力测试报告      ")
    print(f"="*60)
    print(f"性能指标:")
    print(f"  - Join 算法耗时:      {elapsed_time:.4f} 秒")
    print(f"-"*60)
    print(f"Join 结果:")
    print(f"  - 交叉对齐区间数:     {len(joined_partitions):,}")
    print(f"  - 纯噪声桶 (Empty):   {empty_buckets} (占比 {empty_buckets/num_buckets*100:.2f}%)")
    print(f"-"*60)
    print(f"Join Bucket 填充:")
    print(f"  - 总项数 (Real+Dummy):{total_real_items + total_dummy_items:,} 项")
    print(f"  - 真实连接结果 (Real):{total_real_items:,} 项")
    print(f"  - 噪音填充 (Dummy):   {total_dummy_items:,} 项")
    if total_real_items > 0:
        print(f"  - 膨胀系数 (Ratio):   {(total_real_items + total_dummy_items)/total_real_items:.2f}x")
    print(f"  - 平均 Bucket 大小:   {avg_bucket_size:.2f} items")
    
    if total_real_items > 0:
        sample_payload = next((p[0] for bucket in joined_buckets for k,f,p in bucket if k != -999 and len(p)>0), None)
        print(f"  - Payload 样本:       {sample_payload}")
    print(f"="*60 + "\n")


# 3. 单表处理包装器
def process_single_table(name, records, join_key_name, D, eps_part, delta_part, eps_buck, delta_buck, sensitivity):
    """辅助函数：执行单表的 Partition 和 Bucket 流程"""
    partition_algo = PrivatePartition(epsilon=eps_part, delta=delta_part, domain_size=D, sensitivity=sensitivity)
    
    t_start = time.time()
    # 使用通用的提取器获取结构化数据
    sorted_stats = preprocess_table_data(records, join_key_name)
    partitions = partition_algo.run_partition(sorted_stats)
    
    bucket_processor = BucketProcessor(partitions=partitions, epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity)
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    t_end = time.time()
    
    print_summary(name, partitions, final_buckets, t_end - t_start, D)
    return partitions, final_buckets

def generate_uniform_partitions(D, binnum):
    """
    不消耗隐私预算的均匀切分工具。
    将定义域 D 划分为 binnum 个等宽区间。
    """
    partitions = []
    step = D // binnum
    start = 1
    for _ in range(binnum - 1):
        partitions.append((int(start), int(start + step - 1)))
        start += step
    partitions.append((int(start), int(D))) # 最后一个桶兜底
    return partitions

def process_single_table_uniform(name, records, join_key_name, D, partitions, eps_buck, delta_buck, sensitivity):
    """辅助函数：跳过动态 Partition，直接使用传入的静态 partitions 进行 Bucketing"""
    t_start = time.time()
    sorted_stats = preprocess_table_data(records, join_key_name)
    
    bucket_processor = BucketProcessor(partitions=partitions, epsilon=eps_buck, delta=delta_buck, sensitivity=sensitivity)
    final_buckets = bucket_processor.distribute_and_pad(sorted_stats, dummy_key=-999)
    t_end = time.time()
    
    print_summary(name, partitions, final_buckets, t_end - t_start, D)
    return partitions, final_buckets


# 4. 数据生成器
def _generate_clustered_keys(D, num_records, fixed_centers):
    """底层 key 生成器"""
    num_clusters = len(fixed_centers)
    cluster_size = int(num_records * 0.8) // num_clusters
    cluster_data = np.random.normal(loc=fixed_centers[:, None], scale=D*0.0001, size=(num_clusters, cluster_size)).flatten()
    records_noise = num_records - len(cluster_data)
    uniform_data = np.random.randint(1, D+1, size=records_noise)
    all_keys = np.concatenate([cluster_data, uniform_data])
    return np.clip(all_keys, 1, D).astype(int)

def generate_fk_tables(D, N_1, N_2, N_3):
    """生成具有连带外键关系的三张表"""
    print("Generating Foreign Key Tables...")
    t0 = time.time()
    
    # 定义表1和表2在 B 上的热点交集
    centers_B = np.random.randint(1, D, size=50)
    # 定义表2和表3在 C 上的热点交集
    centers_C = np.random.randint(1, D, size=50)
    
    # Table 1: (A, B) -> Join Key 是 B
    keys_1_B = _generate_clustered_keys(D, N_1, centers_B)
    payloads_1 = [{'A': np.random.randint(1, 1000), 'B': int(k)} for k in keys_1_B]
    table_1 = Table(name="Table_1(A,B)", keys=keys_1_B, payloads=payloads_1, sensitivity=1)
    
    # Table 2: (B, C) -> 兼具 B 和 C
    keys_2_B = _generate_clustered_keys(D, N_2, centers_B)
    keys_2_C = _generate_clustered_keys(D, N_2, centers_C)
    payloads_2 = [{'B': int(b), 'C': int(c)} for b, c in zip(keys_2_B, keys_2_C)]
    table_2 = Table(name="Table_2(B,C)", keys=keys_2_B, payloads=payloads_2, sensitivity=1)
    
    # Table 3: (C, D) -> Join Key 是 C
    keys_3_C = _generate_clustered_keys(D, N_3, centers_C)
    payloads_3 = [{'C': int(k), 'D': np.random.randint(1, 1000)} for k in keys_3_C]
    table_3 = Table(name="Table_3(C,D)", keys=keys_3_C, payloads=payloads_3, sensitivity=1)
    
    print(f"Tables Generated. Time: {time.time() - t0:.4f}s")
    return table_1, table_2, table_3

def generate_same_key_tables(D, N_A, N_B, N_C):
    """生成三张共享同一主键 'Key' 的表"""
    centers = np.random.randint(1, D, size=50)
    
    keys_A = _generate_clustered_keys(D, N_A, centers)
    payloads_A = [{'Key': int(k), 'Attr_A': np.random.randint(1, 100)} for k in keys_A]
    table_A = Table(name="Table_A", keys=keys_A, payloads=payloads_A, sensitivity=1)
    
    keys_B = _generate_clustered_keys(D, N_B, centers)
    payloads_B = [{'Key': int(k), 'Attr_B': np.random.randint(1, 100)} for k in keys_B]
    table_B = Table(name="Table_B", keys=keys_B, payloads=payloads_B, sensitivity=1)
    
    keys_C = _generate_clustered_keys(D, N_C, centers)
    payloads_C = [{'Key': int(k), 'Attr_C': np.random.randint(1, 100)} for k in keys_C]
    table_C = Table(name="Table_C", keys=keys_C, payloads=payloads_C, sensitivity=1)
    
    return table_A, table_B, table_C

def generate_4way_same_key_tables(D, N_A, N_B, N_C, N_D):
    """
    生成四张共享同一主键 'Key' 的表，且 Payload 极简（只含 Key）
    """
    centers = np.random.randint(1, D, size=50)
    
    keys_A = _generate_clustered_keys(D, N_A, centers)
    # 极简 Payload：仅保留连接键，消除多余属性的笛卡尔积开销
    payloads_A = [{'Key': int(k)} for k in keys_A]
    table_A = Table(name="Table_A", keys=keys_A, payloads=payloads_A, sensitivity=1)
    
    keys_B = _generate_clustered_keys(D, N_B, centers)
    payloads_B = [{'Key': int(k)} for k in keys_B]
    table_B = Table(name="Table_B", keys=keys_B, payloads=payloads_B, sensitivity=1)
    
    keys_C = _generate_clustered_keys(D, N_C, centers)
    payloads_C = [{'Key': int(k)} for k in keys_C]
    table_C = Table(name="Table_C", keys=keys_C, payloads=payloads_C, sensitivity=1)

    keys_D = _generate_clustered_keys(D, N_D, centers)
    payloads_D = [{'Key': int(k)} for k in keys_D]
    table_D = Table(name="Table_D", keys=keys_D, payloads=payloads_D, sensitivity=1)
    
    return table_A, table_B, table_C, table_D


# 补充一个 2-Way 数据生成器 (可选)
def generate_2way_tables(D, N_A, N_B):
    centers = np.random.randint(1, D, size=50)
    keys_A = _generate_clustered_keys(D, N_A, centers)
    payloads_A = [{'Key': int(k), 'Attr_A': np.random.randint(1, 100)} for k in keys_A]
    table_A = Table(name="Table_A", keys=keys_A, payloads=payloads_A, sensitivity=1)
    
    keys_B = _generate_clustered_keys(D, N_B, centers)
    payloads_B = [{'Key': int(k), 'Attr_B': np.random.randint(1, 100)} for k in keys_B]
    table_B = Table(name="Table_B", keys=keys_B, payloads=payloads_B, sensitivity=1)
    return table_A, table_B


import pandas as pd
def load_real_table(csv_path, table_name, join_key_name, filter_attribute=None, filter_value=None, sensitivity=1):
    """
    从真实的 CSV 文件中加载数据，进行过滤，并转换为 DP 算法所需的 Table 对象。
    
    参数:
    - csv_path: CSV 文件的绝对或相对路径
    - table_name: 表的自定义名称 (用于日志打印区分)
    - join_key_name: 用于后续 Partition/Join 的核心键 (例如 'account_id')
    - filter_attribute: 需要过滤的列名 (例如 'operation'，如果不需过滤填 None)
    - filter_value: 需要过滤的值 (例如 'VYBER KARTOU'，如果不需过滤填 None)
    - sensitivity: 敏感度设置，默认为 1
    
    返回:
    - Table 对象 (兼容你现有的 data_structure.Table)
    """
    print(f"Loading real data for [{table_name}] from {csv_path}...")
    t0 = time.time()
    
    # 1. 读取 CSV (继承之前的排查经验，使用分号分隔)
    try:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}")
        return None
    
    # 2. 按条件过滤数据
    if filter_attribute is not None and filter_value is not None:
        initial_len = len(df)
        df = df[df[filter_attribute] == filter_value]
        print(f"  -> Applied Filter [{filter_attribute} == '{filter_value}']: {initial_len} -> {len(df)} rows")
        
    # 3. 清理无效的 Join Key 
    # (如果 Join Key 是 NaN，说明这条数据无法参与 Join，直接丢弃)
    df = df.dropna(subset=[join_key_name])
    
    # DP Partition 算法通常需要定义域 D 是整数，因此将 Join Key 强制转为 int
    df[join_key_name] = df[join_key_name].astype(int)
    
    # 4. 提取 Payload 和 Keys
    # to_dict('records') 会将 DataFrame 转换为 [{"col1": val1, ...}, {"col1": val2, ...}] 的字典列表
    payloads = df.to_dict('records')
    keys = df[join_key_name].tolist()
    
    # 5. 实例化并返回你定义的 Table 对象
    real_table = Table(name=table_name, keys=keys, payloads=payloads, sensitivity=sensitivity)
    
    print(f"✅ [{table_name}] Successfully generated Table object with {len(payloads)} records. Time: {time.time() - t0:.4f}s")
    return real_table


import time

class ObliviousComplexityLogger:
    """
    用于记录 Oblivious Join 阶段 O(N*M) 真实配对复杂度，
    以及 MPC 下 Resize/Repartition 带来的 Oblivious Sort, Linear Scan, Partition 开销。
    """
    def __init__(self, filepath="oblivious_complexity.log"):
        self.filepath = filepath
        
        # O(N*M) 核心扫描统计
        self.grand_total_pairs = 0
        self.current_join_pairs = 0
        
        # Oblivious 数据结构维护开销统计 (累加处理的元素总数)
        self.total_sort_elements = 0
        self.total_scan_elements = 0
        self.total_partition_elements = 0
        
        self.current_join_name = ""
        
        # 时间记录
        self.global_start_time = time.time()
        self.join_start_time = 0
        
        # 初始化并清空日志文件
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("="*85 + "\n")
            f.write(" Oblivious Multi-Party Computation (MPC) Complexity & Time Log\n")
            f.write("="*85 + "\n\n")

    def start_join(self, join_name: str):
        """标记一个新的 Join 阶段（或路由重切分阶段）开始，并记录开始时间"""
        self.current_join_name = join_name
        self.current_join_pairs = 0
        self.join_start_time = time.time()
        
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n--- 启动 MPC 操作阶段: {join_name} ---\n")

    def log_bucket_pair(self, size_A: int, size_B: int):
        """记录单次物理 Bucket 对的矩阵扫描次数 (O(N*M))"""
        pair_count = size_A * size_B
        self.current_join_pairs += pair_count
        
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"  [Pairing] Bucket A: {size_A:<8} | Bucket B: {size_B:<8} | 扫描次数 (A*B): {pair_count:<12,}\n")

    # =================================================================
    # 新增：Oblivious 数据结构维护开销 (Resize & Repartition)
    # =================================================================
    def log_oblivious_sort(self, size: int, description: str = ""):
        """记录 Oblivious Sort 操作的数组大小"""
        self.total_sort_elements += size
        desc_str = f" | 场景: {description}" if description else ""
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"  [Oblivious Sort]        Size: {size:<10,} {desc_str}\n")
            
    def log_oblivious_linear_scan(self, size: int, description: str = ""):
        """记录 Oblivious Linear Scan 操作的数组大小"""
        self.total_scan_elements += size
        desc_str = f" | 场景: {description}" if description else ""
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"  [Oblivious Linear Scan] Size: {size:<10,} {desc_str}\n")
            
    def log_oblivious_partition(self, size: int, description: str = ""):
        """记录 Oblivious Partition 操作的数组大小 (包含带复杂计算的扫描)"""
        self.total_partition_elements += size
        desc_str = f" | 场景: {description}" if description else ""
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"  [Oblivious Partition]   Size: {size:<10,} {desc_str}\n")

    def end_join(self):
        """结束当前阶段，汇总扫描次数并计算耗时"""
        self.grand_total_pairs += self.current_join_pairs
        elapsed_time = time.time() - self.join_start_time
        
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f">>> [{self.current_join_name}] 阶段结束 | 累计配对扫描次数: {self.current_join_pairs:,} | 耗时: {elapsed_time:.4f} 秒\n\n")

    def end_simulation(self):
        """结束全流程，打印全局汇总结果与总耗时"""
        global_elapsed_time = time.time() - self.global_start_time
        
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*85 + "\n")
            f.write(" 🏆 MPC 全局开销汇总 (Grand Total)\n")
            f.write("="*85 + "\n")
            f.write(f" 1. [O(N*M) 配对扫描] 总比较次数:      {self.grand_total_pairs:,}\n")
            f.write(f" 2. [Oblivious Sort] 累计处理元素总数: {self.total_sort_elements:,}\n")
            f.write(f" 3. [Linear Scan]    累计处理元素总数: {self.total_scan_elements:,}\n")
            f.write(f" 4. [Partition]      累计处理元素总数: {self.total_partition_elements:,}\n")
            f.write(f" ⏱️  端到端总运行耗时 (Wall-clock Time): {global_elapsed_time:.4f} 秒\n")
            f.write("="*85 + "\n")
            
        print(f"\n📊 [MPC 复杂度打点] 详细日志已导出至: {self.filepath}")
        print(f"   - 总配对扫描 (O(N*M)): {self.grand_total_pairs:,} 次")
        print(f"   - Oblivious Sort 总计规模: {self.total_sort_elements:,} 个元素")
        print(f"   - Linear Scan    总计规模: {self.total_scan_elements:,} 个元素")
        print(f"   - Partition      总计规模: {self.total_partition_elements:,} 个元素")
        print(f"   - 模拟器总耗时: {global_elapsed_time:.4f} 秒")