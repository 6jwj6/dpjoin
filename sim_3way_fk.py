import time
from utils import generate_fk_tables, process_single_table, extract_flat_table_from_buckets, print_join_summary
from join_mechanism import DPJoiner, JoinMetadata


def run_foreign_key_join_benchmark(D, N_1, N_2, N_3, eps_total=1.0, delta_total=1e-6):
    """
    外键级联连接: (表1 ⨝ 表2 ON B) ⨝ 表3 ON C
    由于 Join 1 后连接键改变，必须走 Method 1 (重新提取并 Partition)
    """
    print(f"\n{'='*70}")
    print(f"Starting Foreign Key Join: Table1(A,B) ⨝ Table2(B,C) ⨝ Table3(C,D)")
    print(f"Total Budget: eps={eps_total}, delta={delta_total}")
    print(f"{'='*70}")

    global_start_time = time.time() # <--- [新增] 记录全局开始时间
    
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
    parts_12, bucks_12 = joiner_12.run_full_join(parts_1, bucks_1, parts_2, bucks_2, merge_factor=2)
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
    parts_123, bucks_123 = joiner_123.run_full_join(parts_12_new, bucks_12_new, parts_3, bucks_3, merge_factor=3)
    t_join2_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终外键级联连接结果: Table_123(A,B,C,D)")
    print("#"*60)
    print_join_summary(parts_123, bucks_123, t_join2_end - t_join2_start)
    
    global_end_time = time.time() # <--- [新增] 记录全局结束时间
    print(f"\n🚀 [不同 key 3way-join] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")

    return parts_123, bucks_123


if __name__ == "__main__":
    run_foreign_key_join_benchmark(D=1000000, N_1=100000, N_2=80000, N_3=60000)