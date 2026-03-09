import time
import pandas as pd
from utils import (
    load_real_table, process_single_table, print_join_summary, ObliviousComplexityLogger, 
    extract_flat_table_from_buckets, generate_uniform_partitions, process_single_table_uniform
)
from join_mechanism import DPJoiner, JoinMetadata

# =====================================================================
# 路线 A：流式连接 (Pipelined) - 最优精度与自适应
# =====================================================================
def run_q5_3way_join_pipelined(table_A, table_B, table_C, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n [路线 A] Q5 真实数据: 流式连接 (Pipelined)\n{'='*70}")
    global_start_time = time.time()
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 5.0
    delta_stage = (delta_total - delta_meta) / 5.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)
    print(f"[{table_A.name}] Meta 推导: a={meta_A.a}, b={meta_A.b}")
    print(f"[{table_B.name}] Meta 推导: a={meta_B.a}, b={meta_B.b}")
    print(f"[{table_C.name}] Meta 推导: a={meta_C.a}, b={meta_C.b}")

    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table(table_A.name, table_A.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table(table_B.name, table_B.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table(table_C.name, table_C.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    # 初始化 Logger
    logger = ObliviousComplexityLogger(filepath="q5_pipelined_complexity.log")

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    logger.start_join("1st Join: Account ⨝ Trans")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    
    parts_AB, bucks_AB = joiner_AB.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, 
        meta_A=meta_A, meta_B=meta_B, merge_factor=2, 
        logger=logger  # 传入 logger
    )
    logger.end_join()

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans) ⨝ Order ...")
    logger.start_join("2nd Join: AB ⨝ Order")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(
        parts_AB, bucks_AB, parts_C, bucks_C, 
        meta_A=meta_AB, meta_B=meta_C, merge_factor=3, 
        logger=logger  # 传入 logger
    )
    logger.end_join()
    
    # 汇总输出
    logger.end_simulation()
        
    t_j2_s = time.time()
    # 🚀 继续使用 DUB 优化
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB, bucks_AB, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=3)
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 A: 流式连接)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    print(f"\n🚀 [路线 A 总结] 总运行耗时: {time.time() - global_start_time:.4f} 秒\n")


# =====================================================================
# 路线 B：重切分 (Repartitioned)
# =====================================================================
def run_q5_3way_join_repartitioned(table_A, table_B, table_C, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n [路线 B] Q5 真实数据: 重切分 (Repartitioned)\n{'='*70}")
    global_start_time = time.time()
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 6.0
    delta_stage = (delta_total - delta_meta) / 6.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)
    print(f"[{table_A.name}] Meta 推导: a={meta_A.a}, b={meta_A.b}")
    print(f"[{table_B.name}] Meta 推导: a={meta_B.a}, b={meta_B.b}")
    print(f"[{table_C.name}] Meta 推导: a={meta_C.a}, b={meta_C.b}")

    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table(table_A.name, table_A.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table(table_B.name, table_B.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table(table_C.name, table_C.payloads, 'account_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_C.b)

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_j1_s = time.time()
    parts_AB, bucks_AB = joiner_AB.run_full_join(parts_A, bucks_A, parts_B, bucks_B, meta_A=meta_A, meta_B=meta_B, merge_factor=2)
    print_join_summary(parts_AB, bucks_AB, time.time() - t_j1_s)

    print("\n" + "*"*60 + "\n[Phase 2.5] 模拟路由：强制提取重切分 (注意阈值 T 的飙升)\n" + "*"*60)
    records_AB = extract_flat_table_from_buckets(bucks_AB)
    if not records_AB:
        print("警告：Join 1 生成数据为空。")
        return
        
    parts_AB_new, bucks_AB_new = process_single_table(
        "Table_AB(Repart)", records_AB, 'account_id', D, 
        eps_stage, delta_stage, eps_stage, delta_stage, meta_AB.b
    )

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans)_new ⨝ Order ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_j2_s = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB_new, bucks_AB_new, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=3)
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 B: 重切分)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    print(f"\n🚀 [路线 B 总结] 总运行耗时: {time.time() - global_start_time:.4f} 秒\n")


# =====================================================================
# 路线 C：均匀切分 (Uniform)
# =====================================================================
def run_q5_3way_join_uniform(table_A, table_B, table_C, D, binnum, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n [路线 C] Q5 真实数据: 均匀切分 (Uniform) [Binnum={binnum}]\n{'='*70}")
    global_start_time = time.time()
    
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 3.0
    delta_stage = (delta_total - delta_meta) / 3.0

    print("\n[Phase 0] 计算真实表元数据...")
    meta_A = JoinMetadata.from_base_table([r['account_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['account_id'] for r in table_B.payloads], eps_meta, delta_meta)
    meta_C = JoinMetadata.from_base_table([r['account_id'] for r in table_C.payloads], eps_meta, delta_meta)
    print(f"[{table_A.name}] Meta 推导: a={meta_A.a}, b={meta_A.b}")
    print(f"[{table_B.name}] Meta 推导: a={meta_B.a}, b={meta_B.b}")
    print(f"[{table_C.name}] Meta 推导: a={meta_C.a}, b={meta_C.b}")
    
    uniform_parts = generate_uniform_partitions(D, binnum)

    print("\n[Phase 1] 基础表 Uniform Bucketing & Padding...")
    parts_A, bucks_A = process_single_table_uniform(table_A.name+"_Uni", table_A.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table_uniform(table_B.name+"_Uni", table_B.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_B.b)
    parts_C, bucks_C = process_single_table_uniform(table_C.name+"_Uni", table_C.payloads, 'account_id', D, uniform_parts, eps_stage, delta_stage, meta_C.b)

    print("\n[Phase 2] 执行 1st Join: Account ⨝ Trans ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    t_j1_s = time.time()
    parts_AB, bucks_AB = joiner_AB.run_full_join(parts_A, bucks_A, parts_B, bucks_B, meta_A=meta_A, meta_B=meta_B, merge_factor=1)
    print_join_summary(parts_AB, bucks_AB, time.time() - t_j1_s)

    print("\n[Phase 3] 执行 2nd Join: (Account ⨝ Trans) ⨝ Order ...")
    meta_ABC = meta_AB.join(meta_C)
    joiner_ABC = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_ABC.b)
    t_j2_s = time.time()
    parts_ABC, bucks_ABC = joiner_ABC.run_full_join(parts_AB, bucks_AB, parts_C, bucks_C, meta_A=meta_AB, meta_B=meta_C, merge_factor=1)
    
    print("\n" + "#"*60 + "\n 最终 Q5 结果 (路线 C: 均匀切分)\n" + "#"*60)
    print_join_summary(parts_ABC, bucks_ABC, time.time() - t_j2_s)
    print(f"\n🚀 [路线 C 总结] 总运行耗时: {time.time() - global_start_time:.4f} 秒\n")


if __name__ == "__main__":
    DOMAIN_SIZE = 15000   
    BASE_PATH = './financial/'
    
    print(">> 正在加载真实 CSV 数据集并执行条件过滤 (Q5)...")
    
    # 1. 加载 Account 表 (过滤 district_id = 18)
    t_account = load_real_table(
        csv_path=BASE_PATH + 'account.csv', 
        table_name='Account (district=18)', 
        join_key_name='account_id', 
        # filter_attribute='district_id', 
        # filter_value=18, 
        sensitivity=1
    )
    
    # 2. 加载 Trans 表 (过滤 operation = 'VYBER KARTOU')
    t_trans = load_real_table(
        csv_path=BASE_PATH + 'trans.csv', 
        table_name='Trans (operation=VYBER KARTOU)', 
        join_key_name='account_id', 
        filter_attribute='operation', 
        filter_value='VYBER KARTOU',
        sensitivity=1
    )
    
    # 3. 加载 Order 表 (过滤 k_symbol = 'LEASING')
    t_order = load_real_table(
        csv_path=BASE_PATH + 'order.csv', 
        table_name='Order (k_symbol=LEASING)', 
        join_key_name='account_id', 
        # filter_attribute='k_symbol', 
        # filter_value='LEASING',
        sensitivity=1
    )
    
    if t_account and t_trans and t_order and len(t_account.payloads) > 0 and len(t_trans.payloads) > 0 and len(t_order.payloads) > 0:
        print("\n\n" + "/"*80 + "\n 启动 Q5 全套基准测试 (Benchmark)\n" + "/"*80)
        
        # 路线 A: 流式
        run_q5_3way_join_pipelined(t_account, t_trans, t_order, D=DOMAIN_SIZE)
        
        # 路线 B: 重切分
        run_q5_3way_join_repartitioned(t_account, t_trans, t_order, D=DOMAIN_SIZE)
        
        # 路线 C: 均匀网格
        run_q5_3way_join_uniform(t_account, t_trans, t_order, D=DOMAIN_SIZE, binnum=8)
    else:
        print("❌ 数据加载异常或有效记录为空。请检查过滤器！")