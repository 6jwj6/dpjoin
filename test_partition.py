import time
from collections import defaultdict
from utils import generate_2way_tables
from join_mechanism import JoinMetadata


from private_partition import PrivatePartitionOffline 

from private_partition import PrivatePartitionParallel 

def benchmark_pure_partition():
    print(f"\n{'='*70}\n 🚀 纯净版 Partition 算法核心性能基准测试 (Pure Benchmark)\n{'='*70}")
    
    # ==========================================
    # 1. 设定测试规模
    # ==========================================
    DOMAIN_SIZE = 5_000_000
    N_RECORDS = 5_000_000  # 50万条数据
    
    print(f"\n[Phase 0] 正在生成测试数据...")
    print(f"  - 域大小 (Domain Size): {DOMAIN_SIZE}")
    print(f"  - 记录数 (Records): {N_RECORDS}")
    
    t_gen_start = time.time()
    table_A, _ = generate_2way_tables(DOMAIN_SIZE, N_A=N_RECORDS, N_B=10)
    print(f"  - 数据生成耗时: {time.time() - t_gen_start:.2f} 秒")

    # ==========================================
    # 2. 准备 DP 参数与元数据
    # ==========================================
    eps_total = 1.5
    delta_total = 1e-6
    
    eps_meta = eps_total * 0.2
    delta_meta = delta_total * 0.2
    eps_part = (eps_total - eps_meta) / 2.0 
    delta_part = (delta_total - delta_meta) / 2.0

    print("\n[Phase 1] 提取 Key 并准备数据...")
    raw_keys = [r['Key'] for r in table_A.payloads]
    meta_A = JoinMetadata.from_base_table(raw_keys, eps_meta, delta_meta)
    sensitivity = meta_A.b
    sensitivity = 10
    print(f"  - 计算得到的截断敏感度 (Sensitivity b): {sensitivity}")

    results = {}

    # ==========================================
    # 4. 运行串行版本 (Baseline)
    # ==========================================
    print(f"\n{'-'*70}\n ▶ [测试 1] 串行 Partition (Sequential Baseline)\n{'-'*70}")
    
    # 实例化串行算法
    partition_seq = PrivatePartitionOffline(
        epsilon=eps_part, 
        delta=delta_part, 
        domain_size=DOMAIN_SIZE, 
        sensitivity=sensitivity
    )
    
    # 【核心计时区间】
    t0 = time.time()
    parts_seq = partition_seq.run_partition(raw_keys)
    time_seq = time.time() - t0
    
    results['Sequential'] = time_seq
    print(f"  ✓ 串行耗时: {time_seq:.4f} 秒, 产出桶数量: {len(parts_seq)}")

    # ==========================================
    # 5. 运行并行版本 (测试不同的进程数)
    # ==========================================
    
    process_configs = [2, 4, 8, 10, 20] 
    
    for workers in process_configs:
        print(f"\n{'-'*70}\n ▶ [测试] 并行 Partition (Workers = {workers})\n{'-'*70}")
        
        # 实例化并行算法
        partition_par = PrivatePartitionParallel(
            epsilon=eps_part, 
            delta=delta_part, 
            domain_size=DOMAIN_SIZE, 
            num_threads=workers,  # 这里底层已经是多进程了
            sensitivity=sensitivity
        )
        
        # 【核心计时区间】
        t1 = time.time()
        parts_par = partition_par.run_partition(raw_keys)
        time_par = time.time() - t1
        
        results[f'Parallel_{workers}'] = time_par
        print(f"  ✓ 并行耗时: {time_par:.4f} 秒, 产出桶数量: {len(parts_par)}")
        
        # 验证一下产出桶数量是否和串行处于同一量级（证明 DP 切分逻辑没有损坏）
        diff_ratio = abs(len(parts_par) - len(parts_seq)) / max(len(parts_seq), 1)
        if diff_ratio > 0.1:
            print(f"  ⚠️ 警告: 并行版本的桶数量与串行差异较大 ({len(parts_par)} vs {len(parts_seq)})")

    # ==========================================
    # 6. 打印最终性能对比报告
    # ==========================================
    print(f"\n\n{'='*70}\n 📊 纯算法性能对比报告 (Pure Algorithm Benchmark)\n{'='*70}")
    print(f"{'运行模式':<20} | {'耗时 (秒)':<15} | {'加速比 (Speedup)':<15}")
    print("-" * 55)
    
    # 打印串行基准
    print(f"{'Sequential (1x)':<20} | {results['Sequential']:<15.4f} | {'Baseline':<15}")
    
    # 打印并行结果
    for workers in process_configs:
        t_par = results[f'Parallel_{workers}']
        speedup = results['Sequential'] / t_par if t_par > 0 else 0
        print(f"{f'Parallel ({workers} workers)':<20} | {t_par:<15.4f} | {speedup:<15.2f}x")
    
    print("=" * 70)

# =====================================================================
# Windows 和 macOS 的多进程安全入口
# =====================================================================
if __name__ == "__main__":
    benchmark_pure_partition()