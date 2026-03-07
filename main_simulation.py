import numpy as np
import time
from data_structure import Table 
from private_partition import PrivatePartitionOffline as PrivatePartition 
from bucket_mechanism import BucketProcessor
from join_mechanism import DPJoiner, JoinMetadata

# =====================================================================
# 核心辅助函数：状态解包与重组 (Oblivious Routing Simulators)
# =====================================================================

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

# =====================================================================
# 数据生成器：模拟 Foreign Key Schema
# =====================================================================

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

# =====================================================================
# 打印与统计模块
# =====================================================================

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

# =====================================================================
# 核心业务逻辑
# =====================================================================

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

def run_foreign_key_join_benchmark(D, N_1, N_2, N_3, eps_total=1.0, delta_total=1e-6):
    """
    外键级联连接: (表1 ⨝ 表2 ON B) ⨝ 表3 ON C
    由于 Join 1 后连接键改变，必须走 Method 1 (重新提取并 Partition)
    """
    print(f"\n{'='*70}")
    print(f"Starting Foreign Key Join: Table1(A,B) ⨝ Table2(B,C) ⨝ Table3(C,D)")
    print(f"Total Budget: eps={eps_total}, delta={delta_total}")
    print(f"{'='*70}")

    # --- 1. 并行组合下的预算分配 ---
    # 这是一条串行流水线，任意一条数据最多经历以下阶段：
    # Meta推导(1) -> 本表Part+Buck(2) -> Join1(1) -> 表12重组Part+Buck(2) -> Join2(1)
    # 共 7 个动作阶段。为了保证成功率，我们依然重仓 Meta。
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    eps_stage = (eps_total - eps_meta) / 6.0
    delta_stage = (delta_total - delta_meta) / 6.0

    # 生成数据
    t1, t2, t3 = generate_fk_tables(D, N_1, N_2, N_3)
    
    # [Phase 0] Meta 计算
    print("\n[Phase 0] 计算基础表元数据 (Noisy Max Frequency)...")
    
    # 表1: 只有 B 作为连接键
    meta_1 = JoinMetadata.from_base_table([r['B'] for r in t1.payloads], eps_meta, delta_meta)
    
    # -----------------------------------------------------------------
    # 【核心修正】：表2 作为桥接表，既要连 B 又要连 C
    # 提取两列，并调用新的 multiple_keys 方法，取两者的共同噪声上限
    # -----------------------------------------------------------------
    keys_2_B = [r['B'] for r in t2.payloads]
    keys_2_C = [r['C'] for r in t2.payloads]
    meta_2 = JoinMetadata.from_base_table_multiple_keys([keys_2_B, keys_2_C], eps_meta, delta_meta)
    
    # 表3: 只有 C 作为连接键
    meta_3 = JoinMetadata.from_base_table([r['C'] for r in t3.payloads], eps_meta, delta_meta)

    print(f"Table 1 Meta (on B):   a={meta_1.a}, b={meta_1.b}")
    print(f"Table 2 Meta (on B,C): a={meta_2.a}, b={meta_2.b} (取两键中较大者)")
    print(f"Table 3 Meta (on C):   a={meta_3.a}, b={meta_3.b}")

    # [Phase 1] 表1 和 表2 针对 Key B 进行 Partition & Padding
    print("\n[Phase 1] 基础表 1 和 2 针对 Key 'B' 预处理...")
    parts_1, bucks_1 = process_single_table("Table_1(A,B)", t1.payloads, 'B', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_1.b)
    parts_2, bucks_2 = process_single_table("Table_2(B,C)", t2.payloads, 'B', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_2.b)

    # [Phase 2] 1st Join (ON B)
    print("\n[Phase 2] 执行 1st Join: Table_1 ⨝ Table_2 ON B ...")
    meta_12 = meta_1.join(meta_2) # b 会发生膨胀
    joiner_12 = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_12.b)
    
    t_join1_start = time.time()
    parts_12, bucks_12 = joiner_12.run_join(parts_1, bucks_1, parts_2, bucks_2, merge_factor=2)
    t_join1_end = time.time()
    print_join_summary(parts_12, bucks_12, t_join1_end - t_join1_start)

    # ==========================================================
    # [Phase 2.5] 核心路由阶段：提取中间结果，切换 Key 至 C
    # ==========================================================
    print("\n" + "*"*60)
    print("[Phase 2.5] 状态提取与路由：切换 Join Key 至 'C'")
    print("*"*60)
    
    # 提取扁平化的 records: 形如 [{'A':a, 'B':b, 'C':c}, ...]
    records_12 = extract_flat_table_from_buckets(bucks_12)
    print(f"成功从 Join 1 中提取出 {len(records_12):,} 条有效载荷记录。")
    
    if len(records_12) == 0:
        print("警告：Join 1 生成的数据为空，后续流程终止。")
        return
        
    # 对提取出的 records_12 针对新 Key 'C' 执行重切分和重装桶
    # 注意：此时使用的敏感度是膨胀后的 meta_12.b
    parts_12_new, bucks_12_new = process_single_table(
        "Table_12(A,B,C) ON C", records_12, 'C', D, 
        eps_stage, delta_stage, eps_stage, delta_stage, meta_12.b
    )

    # ==========================================================
    # [Phase 3] 2nd Join (ON C)
    # ==========================================================
    print("\n[Phase 3] 表3 针对 Key 'C' 预处理...")
    parts_3, bucks_3 = process_single_table("Table_3(C,D)", t3.payloads, 'C', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_3.b)

    print("\n[Phase 4] 执行 2nd Join: Table_12 ⨝ Table_3 ON C ...")
    # 推导最终的敏感度
    meta_123 = meta_12.join(meta_3)
    
    joiner_123 = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_123.b)
    t_join2_start = time.time()
    parts_123, bucks_123 = joiner_123.run_join(parts_12_new, bucks_12_new, parts_3, bucks_3, merge_factor=3)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终外键级联连接结果: Table_123(A,B,C,D)")
    print("#"*60)
    print_join_summary(parts_123, bucks_123, t_join2_end - t_join2_start)
    
    return parts_123, bucks_123

# =====================================================================
# 函数 1：不进行 Repartition 的流式 Join (Method 2 - 路线 A)
# =====================================================================
def run_same_key_join_pipelined(table_A, table_B, table_C, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}")
    print(f" [路线 A] 启动 Same-Key Join (不重新切分 / Pipelined)")
    print(f"{'='*70}")

    global_start_time = time.time() # <--- [新增] 记录全局开始时间
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    eps_stage = (eps_total - eps_meta) / 4.0
    delta_stage = (delta_total - delta_meta) / 4.0

    # [Phase 0] Meta 计算
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['Key'] for r in table_C.payloads], eps_meta, delta_meta)

    # [Phase 1] 基础表预处理
    parts_A, bucks_A = process_single_table("Table_A", table_A.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table("Table_B", table_B.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table("Table_C", table_C.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    # [Phase 2] 1st Join (A ⨝ B)
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_join1_start = time.time()
    parts_AB, bucks_AB = joiner_AB.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
    t_join1_end = time.time()
    print("\n" + "#"*60)
    print(" 1st 流式连接结果 (路线 A: 无 Repartition)")
    print("#"*60)
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # [Phase 3] 2nd Join ((A ⨝ B) ⨝ C) - 【核心：直接沿用 parts_AB 和 bucks_AB】
    print("\n[Phase 3] 流式直连 2nd Join ((A ⨝ B) ⨝ C) ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_join(parts_AB, bucks_AB, parts_C, bucks_C, merge_factor=3)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终流式连接结果 (路线 A: 无 Repartition)")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)

    global_end_time = time.time() # <--- [新增] 记录全局结束时间
    print(f"\n🚀 [路线 A 总结] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")

# =====================================================================
# 函数 2：进行 Repartition 的重构 Join (Method 1 - 路线 B)
# =====================================================================
def run_same_key_join_repartitioned(table_A, table_B, table_C, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}")
    print(f" [路线 B] 启动 Same-Key Join (强制重新切分 / Repartitioned)")
    print(f"{'='*70}")

    global_start_time = time.time() # <--- [新增] 记录全局开始时间

    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    # 相比于流式，这条路线多了重新 Part 和 重装 Buck 两个极其消耗预算的步骤
    eps_stage = (eps_total - eps_meta) / 6.0
    delta_stage = (delta_total - delta_meta) / 6.0

    # [Phase 0] Meta 计算
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['Key'] for r in table_C.payloads], eps_meta, delta_meta)

    # [Phase 1] 基础表预处理
    parts_A, bucks_A = process_single_table("Table_A", table_A.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table("Table_B", table_B.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table("Table_C", table_C.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    # [Phase 2] 1st Join (A ⨝ B)
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_join1_start = time.time()
    parts_AB, bucks_AB = joiner_AB.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
    t_join1_end = time.time()
    print("\n" + "#"*60)
    print(" 1st 重构连接结果 (路线 B: 包含 Repartition)")
    print("#"*60)
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # [Phase 2.5] 提取并强制重新切分
    print("\n" + "*"*60)
    print("[Phase 2.5] 模拟路由：提取明文，使用膨胀后的敏感度强制重切分")
    print("*"*60)
    records_AB = extract_flat_table_from_buckets(bucks_AB)
    
    if len(records_AB) == 0:
        print("警告：Join 1 生成的数据为空，后续流程终止。")
        return
        
    # 【核心灾难点】：这里传入的敏感度是 meta_AB.b，会导致极高的门槛 T
    parts_AB_new, bucks_AB_new = process_single_table(
        "Table_AB(Repart)", records_AB, 'Key', D, 
        eps_stage, delta_stage, eps_stage, delta_stage, meta_AB.b
    )

    # [Phase 3] 2nd Join ((A ⨝ B)_new ⨝ C)
    print("\n[Phase 3] 重构后执行 2nd Join ((A ⨝ B)_new ⨝ C) ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_join(parts_AB_new, bucks_AB_new, parts_C, bucks_C, merge_factor=3)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终重构连接结果 (路线 B: 包含 Repartition)")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)

    global_end_time = time.time() # <--- [新增] 记录全局结束时间
    print(f"\n🚀 [路线 B 总结] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")

# =====================================================================
# 函数 3：均匀网格划分 Join (Method 3- 路线 C)
# =====================================================================
def run_same_key_join_uniform(table_A, table_B, table_C, D, binnum, eps_total=1.0, delta_total=1e-6):
    """
    [路线 C] 均匀网格划分 (Uniform / Equi-width Partitioning)
    基于静态的 binnum 划分区间，不消耗 Partition 隐私预算。
    """
    print(f"\n{'='*70}")
    print(f" [路线 C] 启动 Same-Key Join (均匀网格 / Uniform Partitioned)")
    print(f" 静态桶数量 (Binnum): {binnum}")
    print(f"{'='*70}")

    global_start_time = time.time() # <--- [新增] 记录全局开始时间

    # 【预算分配优势】：因为 Partition 是数据无关的，消耗 0 预算！
    # 只需要把剩余预算分给 Bucketing、Join 1 和 Join 2 这三个阶段即可。
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    eps_stage = (eps_total - eps_meta) / 3.0   # 之前是 /4.0 或 /6.0，现在预算更充裕了！
    delta_stage = (delta_total - delta_meta) / 3.0

    # 生成静态区间
    uniform_parts = generate_uniform_partitions(D, binnum)

    # [Phase 0] Meta 计算
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['Key'] for r in table_C.payloads], eps_meta, delta_meta)

    # [Phase 1] 基础表直接 Bucketing (共用一套 uniform_parts)
    print("\n[Phase 1] 基础表 Uniform Bucketing & Padding...")
    parts_A, bucks_A = process_single_table_uniform("Table_A(Uniform)", table_A.payloads, 'Key', D, uniform_parts, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table_uniform("Table_B(Uniform)", table_B.payloads, 'Key', D, uniform_parts, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table_uniform("Table_C(Uniform)", table_C.payloads, 'Key', D, uniform_parts, eps_stage, delta_stage, meta_C.b)

    # [Phase 2] 1st Join (A ⨝ B)
    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_join1_start = time.time()
    # 对于 Uniform，由于物理区间已经完全相同，其实 merge_factor 不会发挥太大"自适应"作用
    # 但我们仍保持原有的合并逻辑
    parts_AB, bucks_AB = joiner_AB.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=1)
    t_join1_end = time.time()
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # [Phase 3] 2nd Join ((A ⨝ B) ⨝ C)
    print("\n[Phase 3] 执行 2nd Join ((A ⨝ B) ⨝ C) ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_join(parts_AB, bucks_AB, parts_C, bucks_C, merge_factor=1)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终均匀连接结果 (路线 C: Uniform Partitioning)")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)

    global_end_time = time.time() # <--- [新增] 记录全局结束时间
    print(f"\n🚀 [路线 C 总结] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")

if __name__ == "__main__":
    DOMAIN_SIZE = 100_0000   
    NUM_RECORDS_A = 1_000_00
    NUM_RECORDS_B = 80_000    
    NUM_RECORDS_C = 60_000  
    
    # 缩小了一点数据规模以便你可以快速跑完全流程并观察终端打印的 Payload 样本
    # run_foreign_key_join_benchmark(
    #     D=DOMAIN_SIZE, 
    #     N_1=NUM_RECORDS_A, 
    #     N_2=NUM_RECORDS_B,
    #     N_3=NUM_RECORDS_C,
    #     eps_total=1.0,
    #     delta_total=1e-6
    # )

    # 为了保证两个实验完全公平，我们提前生成好一套数据，传给两个函数
    tA, tB, tC = generate_same_key_tables(DOMAIN_SIZE, NUM_RECORDS_A, NUM_RECORDS_B, NUM_RECORDS_C)
    
    # 实验 A：流式 (你会看到非常健康的 10x 左右的膨胀率)
    run_same_key_join_pipelined(tA, tB, tC, DOMAIN_SIZE, eps_total=1.0, delta_total=1e-6)
    
    print("\n\n" + "/"*80 + "\n\n")
    
    # 实验 B：重切分 (你会亲眼看到 T 飙升，桶数量锐减，最终 Dummy 爆炸)
    run_same_key_join_repartitioned(tA, tB, tC, DOMAIN_SIZE, eps_total=1.0, delta_total=1e-6)

    print("\n\n" + "/"*80 + "\n\n")
    
    # 实验 C：均匀网格划分 (我们尝试切分出与路线 A 接近的桶数量，比如 200 个)
    run_same_key_join_uniform(tA, tB, tC, DOMAIN_SIZE, binnum=200, eps_total=1.0, delta_total=1e-6)