import time
from utils import generate_4way_same_key_tables, process_single_table, print_join_summary
from join_mechanism import DPJoiner, JoinMetadata

def run_4way_same_key_pipelined(table_A, table_B, table_C, table_D, D, eps_total=1.0, delta_total=1e-6):
    """
    执行 4-Way 流式连接 (Pipelined Join: A ⨝ B ⨝ C ⨝ D)
    """
    print(f"\n{'='*70}")
    print(f" 启动 4-Way Same-Key Join (流式连接 / Pipelined)")
    print(f"{'='*70}")
    
    global_start_time = time.time()

    # 【预算分配】：依然重仓 Meta (50%)
    # 剩下的 50% 均分给 4个基础表处理 + 3次Join操作 = 7个阶段
    ratio_meta = 0.80 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    eps_stage = (eps_total - eps_meta) / 7.0
    delta_stage = (delta_total - delta_meta) / 7.0

    # [Phase 0] Meta 计算
    print("\n[Phase 0] 计算基础表元数据...")
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['Key'] for r in table_C.payloads], eps_meta, delta_meta)
    meta_D = JoinMetadata.from_base_table([r['Key'] for r in table_D.payloads], eps_meta, delta_meta)

    # [Phase 1] 基础表预处理
    print("\n[Phase 1] 基础表 Partition & Padding...")
    parts_A, bucks_A = process_single_table("Table_A", table_A.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table("Table_B", table_B.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table("Table_C", table_C.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)
    parts_D, bucks_D = process_single_table("Table_D", table_D.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_D.b)

    # [Phase 2] 1st Join (A ⨝ B)
    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_join1_start = time.time()
    # 注意这里使用的是你优化的 run_join (双指针版本)
    parts_AB, bucks_AB = joiner_AB.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
    t_join1_end = time.time()
    print_join_summary(parts_AB, bucks_AB, t_join1_end - t_join1_start)

    # [Phase 3] 2nd Join ((A ⨝ B) ⨝ C)
    print("\n[Phase 3] 流式直连 2nd Join: (A ⨝ B) ⨝ Table_C ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_join2_start = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_join(parts_AB, bucks_AB, parts_C, bucks_C, merge_factor=3)
    t_join2_end = time.time()
    print_join_summary(parts_ABC, bucks_ABC, t_join2_end - t_join2_start)

    # [Phase 4] 3rd Join (((A ⨝ B) ⨝ C) ⨝ D)
    print("\n[Phase 4] 流式直连 3rd Join: (A ⨝ B ⨝ C) ⨝ Table_D ...")
    meta_ABCD = meta_ABC.join(meta_D)
    joiner_ABCD = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABCD.b)
    t_join3_start = time.time()
    parts_ABCD, bucks_ABCD = joiner_ABCD.run_join(parts_ABC, bucks_ABC, parts_D, bucks_D, merge_factor=4)
    t_join3_end = time.time()
    print_join_summary(parts_ABCD, bucks_ABCD, t_join3_end - t_join3_start)
    
    global_end_time = time.time()
    print(f"\n🚀 [4-Way 流式总结] 端到端总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")

if __name__ == "__main__":
    DOMAIN_SIZE = 1_000_000   
    tA, tB, tC, tD = generate_4way_same_key_tables(DOMAIN_SIZE, 100000, 80000, 60000, 50000)
    run_4way_same_key_pipelined(tA, tB, tC, tD, D=DOMAIN_SIZE, eps_total=1.0, delta_total=1e-6)