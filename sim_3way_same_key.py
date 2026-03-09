import time
from utils import generate_same_key_tables, process_single_table, process_single_table_uniform, generate_uniform_partitions, extract_flat_table_from_buckets, print_join_summary
from join_mechanism import DPJoiner, JoinMetadata

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
    parts_AB, bucks_AB = joiner_AB.run_full_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
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
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB, bucks_AB, parts_C, bucks_C, merge_factor=3)
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
    parts_AB, bucks_AB = joiner_AB.run_full_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
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
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB_new, bucks_AB_new, parts_C, bucks_C, merge_factor=3)
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
    parts_AB, bucks_AB = joiner_AB.run_full_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=1)
    t_join1_end = time.time()
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # [Phase 3] 2nd Join ((A ⨝ B) ⨝ C)
    print("\n[Phase 3] 执行 2nd Join ((A ⨝ B) ⨝ C) ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB, bucks_AB, parts_C, bucks_C, merge_factor=1)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终均匀连接结果 (路线 C: Uniform Partitioning)")
    print("#"*60)
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)

    global_end_time = time.time() # <--- [新增] 记录全局结束时间
    print(f"\n🚀 [路线 C 总结] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")


if __name__ == "__main__":
    DOMAIN_SIZE = 1_000_00   
    tA, tB, tC = generate_same_key_tables(DOMAIN_SIZE, 10000, 8000, 6000)
    
    run_same_key_join_pipelined(tA, tB, tC, DOMAIN_SIZE)
    run_same_key_join_repartitioned(tA, tB, tC, DOMAIN_SIZE)
    run_same_key_join_uniform(tA, tB, tC, DOMAIN_SIZE, binnum=8)