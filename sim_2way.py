import time
from utils import (
    generate_2way_tables, 
    process_single_table, 
    print_join_summary,
    process_single_table_uniform,   # 新增引入
    generate_uniform_partitions,    # 新增引入
    ObliviousComplexityLogger       # 新增引入，用于打点 O(N*M) 复杂度
)
from join_mechanism import DPJoiner, JoinMetadata

# =====================================================================
# 路线 A：动态流式连接 (Pipelined DP Partition)
# =====================================================================
def run_2way_join_pipelined(table_A, table_B, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n 🚀 [路线 A] 启动 2-Way Join (Pipelined DP Partition)\n{'='*70}")
    
    # 启用复杂度日志器
    logger = ObliviousComplexityLogger(filepath="2way_pipelined_complexity.log")
    
    # 预算分配修正：
    # 50% 给 Meta。剩下 50% 需要分给：Part_A, Buck_A, Part_B, Buck_B, Join_AB，共 5 份
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 5.0 
    delta_stage = (delta_total - delta_meta) / 5.0

    print("\n[Phase 0] 计算基础表元数据...")
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)

    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table("Table_A", table_A.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table("Table_B", table_B.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)

    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    joiner = DPJoiner(eps_stage, delta_stage, meta_AB.b)
    
    logger.start_join("Table_A ⨝ Table_B (Pipelined)")
    # 完美接入 DUB 截断 (meta_A, meta_B) 和 Logger
    parts_AB, bucks_AB = joiner.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, 
        meta_A=meta_A, meta_B=meta_B, merge_factor=2, logger=logger
    )
    logger.end_join()
    
    print_join_summary(parts_AB, bucks_AB, time.time() - logger.global_start_time)
    logger.end_simulation()

# =====================================================================
# 路线 C：均匀静态网格切分 (Uniform Partition)
# =====================================================================
def run_2way_join_uniform(table_A, table_B, D, binnum, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n 🚀 [路线 C] 启动 2-Way Join (Uniform Partition, binnum={binnum})\n{'='*70}")
    
    logger = ObliviousComplexityLogger(filepath="2way_uniform_complexity.log")
    
    # 预算分配优势：
    # Partition 阶段数据无关，耗费 0 预算。
    # 剩下 50% 只需分给：Buck_A, Buck_B, Join_AB，共 3 份（每个阶段分得的预算更多，加噪更少！）
    ratio_meta = 0.50 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    eps_stage = (eps_total - eps_meta) / 3.0 
    delta_stage = (delta_total - delta_meta) / 3.0

    print("\n[Phase 0] 计算基础表元数据...")
    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)

    # 静态生成均匀网格
    uniform_parts = generate_uniform_partitions(D, binnum)

    print("\n[Phase 1] 基础表 Uniform Bucketing & Padding...")
    parts_A, bucks_A = process_single_table_uniform("Table_A(Uni)", table_A.payloads, 'Key', D, uniform_parts, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table_uniform("Table_B(Uni)", table_B.payloads, 'Key', D, uniform_parts, eps_stage, delta_stage, meta_B.b)

    print("\n[Phase 2] 执行 1st Join: Table_A ⨝ Table_B ...")
    meta_AB = meta_A.join(meta_B)
    joiner = DPJoiner(eps_stage, delta_stage, meta_AB.b)
    
    logger.start_join("Table_A ⨝ Table_B (Uniform)")
    # Uniform 切分由于物理区间严格对齐，merge_factor 固定为 1 即可
    parts_AB, bucks_AB = joiner.run_full_join(
        parts_A, bucks_A, parts_B, bucks_B, 
        meta_A=meta_A, meta_B=meta_B, merge_factor=1, logger=logger
    )
    logger.end_join()
    
    print_join_summary(parts_AB, bucks_AB, time.time() - logger.global_start_time)
    logger.end_simulation()


if __name__ == "__main__":
    DOMAIN_SIZE = 1_000_000
    
    print(">> 正在生成 2-Way 模拟数据...")
    tA, tB = generate_2way_tables(DOMAIN_SIZE, N_A=100_000, N_B=80_000)
    
    # 1. 运行 Pipelined 版本
    run_2way_join_pipelined(tA, tB, DOMAIN_SIZE, eps_total=1.5, delta_total=1e-6)
    
    print("\n\n" + r"/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/" + "\n\n")
    
    # 2. 运行 Uniform 版本
    # 我们可以尝试设置 50 个桶，观察在倾斜数据下它会导致多大的计算不平衡
    run_2way_join_uniform(tA, tB, DOMAIN_SIZE, binnum=8, eps_total=1.5, delta_total=1e-6)