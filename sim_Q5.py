import time
import pandas as pd
from utils import (
    load_real_table, process_single_table, print_join_summary,
    extract_flat_table_from_buckets, generate_uniform_partitions, process_single_table_uniform,
    ObliviousComplexityLogger  # 引入我们写好的日志记录器
)
from join_mechanism import DPJoiner, JoinMetadata

# =====================================================================
# 路线 A：流式连接 (Pipelined) - 最优精度与自适应
# =====================================================================
def run_q5_3way_join_pipelined(table_A, table_B, table_C, D, eps_total=1.5, delta_total=1e-5):
    print(f"\n{'='*70}\n [路线 A] Q5 真实数据: 流式连接 (Pipelined)\n{'='*70}")
    
    logger = ObliviousComplexityLogger(filepath="q5_pipelined_complexity.log")
    
    ratio_meta = 0.65
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 5.0
    delta_stage = (delta_total - delta_meta) / 5.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)

    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table(table_A.name, table_A.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table(table_B.name, table_B.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table(table_C.name, table_C.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    logger.start_join("1st Join: Account ⨝ Trans")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_j1_s = time.time()
    parts_AB, bucks_AB = joiner_AB.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, meta_A=meta_A, meta_B=meta_B, merge_factor=2, logger=logger
    )
    logger.end_join()
    print_join_summary(parts_AB, bucks_AB, time.time() - t_j1_s)

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans) ⨝ Order ...")
    logger.start_join("2nd Join: (Account ⨝ Trans) ⨝ Order")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_j2_s = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(
        parts_AB, bucks_AB, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=3, logger=logger
    )
    logger.end_join()
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 A: 流式连接)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    logger.end_simulation()


# =====================================================================
# 路线 B：重切分 (Repartitioned)
# =====================================================================
def run_q5_3way_join_repartitioned(table_A, table_B, table_C, D, eps_total=1.5, delta_total=1e-5):
    print(f"\n{'='*70}\n [路线 B] Q5 真实数据: 重切分 (Repartitioned)\n{'='*70}")
    
    logger = ObliviousComplexityLogger(filepath="q5_repartitioned_complexity.log")
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 6.0
    delta_stage = (delta_total - delta_meta) / 6.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)

    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table(table_A.name, table_A.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table(table_B.name, table_B.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table(table_C.name, table_C.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    logger.start_join("1st Join: Account ⨝ Trans")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_j1_s = time.time()
    parts_AB, bucks_AB = joiner_AB.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, meta_A=meta_A, meta_B=meta_B, merge_factor=2, logger=logger
    )
    logger.end_join()
    print_join_summary(parts_AB, bucks_AB, time.time() - t_j1_s)

    # =================================================================
    # [Phase 2.5] 模拟路由：强制提取重切分 
    # =================================================================
    print("\n" + "*"*60 + "\n[Phase 2.5] 模拟路由：强制提取重切分\n" + "*"*60)
    
    # 提取有效明文（在 MPC 中这步是密态的）
    records_AB = extract_flat_table_from_buckets(bucks_AB)
    if not records_AB:
        print("警告：Join 1 生成数据为空。")
        return
    
    # 【核心测算】：N_in 是重切分前的总物理数组大小（包含 Dummy）
    N_in = sum(sum(len(p) for k, f, p in b) for b in bucks_AB)
    
    logger.start_join("Phase 2.5: Repartition AB")
    
    parts_AB_new, bucks_AB_new = process_single_table(
        "Table_AB(Repart)", records_AB, 'account_id', D, 
        eps_stage, delta_stage, eps_stage, delta_stage, meta_AB.b
    )
    
    # 【核心测算】：B_total 是重切分后的目标总物理数组大小
    B_total = sum(sum(len(p) for k, f, p in b) for b in bucks_AB_new)
    
    # 🚀🚀🚀 按照你的思维导图，完美植入 Repartition 开销打点 🚀🚀🚀
    logger.log_oblivious_partition(N_in, "Repartition: Generate Noise & Mark")
    logger.log_oblivious_sort(N_in, "Repartition: Sort 1 (Prepare for placement)")
    logger.log_oblivious_sort(N_in + B_total, "Repartition: Sort 2 (Bin Placement)")
    logger.log_oblivious_linear_scan(B_total, "Repartition: Dummy filling scan")
    logger.end_join()
    # =================================================================

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans)_new ⨝ Order ...")
    logger.start_join("2nd Join: (Account ⨝ Trans)_new ⨝ Order")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_j2_s = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(
        parts_AB_new, bucks_AB_new, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=3, logger=logger
    )
    logger.end_join()
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 B: 重切分)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    logger.end_simulation()


# =====================================================================
# 路线 C：均匀切分 (Uniform)
# =====================================================================
def run_q5_3way_join_uniform(table_A, table_B, table_C, D, binnum, eps_total=1.5, delta_total=1e-6):
    print(f"\n{'='*70}\n [路线 C] Q5 真实数据: 均匀切分 (Uniform) [Binnum={binnum}]\n{'='*70}")
    
    logger = ObliviousComplexityLogger(filepath="q5_uniform_complexity.log")
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 3.0
    delta_stage = (delta_total - delta_meta) / 3.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)

    uniform_parts = generate_uniform_partitions(D, binnum)

    print("\n[Phase 1] 基础表 Uniform Bucketing & Padding...")
    parts_A, bucks_A = process_single_table_uniform(table_A.name+"_Uni", table_A.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table_uniform(table_B.name+"_Uni", table_B.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table_uniform(table_C.name+"_Uni", table_C.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_C.b)

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    logger.start_join("1st Join: Account ⨝ Trans")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_j1_s = time.time()
    parts_AB, bucks_AB = joiner_AB.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, meta_A=meta_A, meta_B=meta_B, merge_factor=1, logger=logger
    )
    logger.end_join()
    print_join_summary(parts_AB, bucks_AB, time.time() - t_j1_s)

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans) ⨝ Order ...")
    logger.start_join("2nd Join: (Account ⨝ Trans) ⨝ Order")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_j2_s = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(
        parts_AB, bucks_AB, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=1, logger=logger
    )
    logger.end_join()
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 C: 均匀切分)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    logger.end_simulation()


if __name__ == "__main__":
    DOMAIN_SIZE = 15000   
    BASE_PATH = './financial/'
    
    print(">> 正在加载真实 CSV 数据集并执行条件过滤 (Q5)...")
    
    t_account = load_real_table(csv_path=BASE_PATH + 'account.csv', table_name='Account', join_key_name='account_id',
                                #  filter_attribute='district_id', filter_value=18
                                 )
    t_trans = load_real_table(csv_path=BASE_PATH + 'trans.csv', table_name='Trans', join_key_name='account_id',
                               filter_attribute='operation', filter_value='VYBER KARTOU'
                               )
    t_order = load_real_table(csv_path=BASE_PATH + 'order.csv', table_name='Order', join_key_name='account_id',
                            #    filter_attribute='k_symbol', filter_value='LEASING'
                               )
    
    if t_account and t_trans and t_order and len(t_account.payloads) > 0 and len(t_trans.payloads) > 0 and len(t_order.payloads) > 0:
        print("\n\n" + "/"*80 + "\n 启动 Q5 全套基准测试 (Benchmark)\n" + "/"*80)
        
        # run_q5_3way_join_pipelined(t_account, t_trans, t_order, D=DOMAIN_SIZE)
        run_q5_3way_join_repartitioned(t_account, t_trans, t_order, D=DOMAIN_SIZE)
        # run_q5_3way_join_uniform(t_account, t_trans, t_order, D=DOMAIN_SIZE, binnum=8)
    else:
        print("❌ 数据加载异常或有效记录为空。请检查过滤器！")