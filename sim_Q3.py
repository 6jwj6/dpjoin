import time
# 引入我们之前重构好的工具库和 Join 引擎
from utils import load_real_table, process_single_table, print_join_summary
from join_mechanism import DPJoiner, JoinMetadata

def run_q3_2way_join(table_A, table_B, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}")
    print(f" 🚀 启动真实数据 Q3 模拟: Client ⨝ Disp ON client_id")
    print(f" 预算分配: eps={eps_total}, delta={delta_total}")
    print(f"{'='*70}")
    
    global_start_time = time.time()
    
    # 1. 预算分配
    # 对于 2-way join：50%给 Meta。剩下 50% 均分给 3 个核心阶段 (Part A, Part B, Join AB)
    ratio_meta = 0.80 
    eps_meta = eps_total * ratio_meta
    delta_meta = delta_total * ratio_meta
    
    eps_stage = (eps_total - eps_meta) / 3.0
    delta_stage = (delta_total - delta_meta) / 3.0

    # [Phase 0] Meta 计算
    print("\n[Phase 0] 计算真实表元数据 (Noisy Max Frequency)...")
    meta_A = JoinMetadata.from_base_table([r['client_id'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['client_id'] for r in table_B.payloads], eps_meta, delta_meta)
    
    print(f"[{table_A.name}] Meta 推导: a={meta_A.a}, b={meta_A.b}")
    print(f"[{table_B.name}] Meta 推导: a={meta_B.a}, b={meta_B.b}")

    # [Phase 1] 基础表预处理
    print("\n[Phase 1] 基础表 DP Partition & Padding...")
    parts_A, bucks_A = process_single_table(table_A.name, table_A.payloads, 'client_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table(table_B.name, table_B.payloads, 'client_id', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)

    # [Phase 2] 1st Join (A ⨝ B)
    print("\n[Phase 2] 执行 2-Way Join: Client ⨝ Disp ...")
    meta_AB = meta_A.join(meta_B)
    joiner_AB = DPJoiner(epsilon=eps_stage, delta=delta_stage, sensitivity=meta_AB.b)
    
    t_join_start = time.time()
    # 这里调用你保留了双指针优化的 run_join
    # 如果你想测试极其严谨的 Oblivious O(N*M) 性能，可以改为调用 run_full_join
    parts_AB, bucks_AB = joiner_AB.run_join(
        parts_A, bucks_A, parts_B, bucks_B, 
        meta_A=meta_A, meta_B=meta_B, 
        merge_factor=2
    )
    t_join_end = time.time()
    
    print("\n" + "#"*60)
    print(" 最终 Q3 2-Way 连接结果 (Client ⨝ Disp)")
    print("#"*60)
    print_join_summary(parts_AB, bucks_AB, t_join_end - t_join_start)

    global_end_time = time.time()
    print(f"\n🚀 [Q3 端到端总结] 总运行耗时: {global_end_time - global_start_time:.4f} 秒\n")
    
    return parts_AB, bucks_AB

if __name__ == "__main__":
    # 【定义域大小设置】
    # PKDD'99 数据集中 client_id 最大在 15000 左右，这里设置 20000 留有余量。
    # 设置合理的 D 极大有助于提升性能和降低阈值 T。
    DOMAIN_SIZE = 15000   
    
    print(">> 正在加载真实 CSV 数据集并执行条件过滤...")
    
    # 数据集所在路径
    BASE_PATH = './financial/'

    # 注意 1: Pandas 可能会把 csv 里的 district_id 读成 int 形式，所以过滤值用 18。
    # 如果实际报错说找不到数据，可以尝试改为 filter_value='18'
    t_client = load_real_table(
        csv_path=BASE_PATH +'client.csv', 
        table_name='Client (district=18)', 
        join_key_name='client_id', 
        # filter_attribute='district_id',
        # filter_value=18, 
        sensitivity=1
    )
    
    # 注意 2: Disp 表的 type 列是字符串
    t_disp = load_real_table(
        csv_path=BASE_PATH +'disp.csv', 
        table_name='Disp (type=DISPONENT)', 
        join_key_name='client_id', 
        # filter_attribute='type', 
        # filter_value='DISPONENT',
        sensitivity=1
    )
    
    if t_client is not None and t_disp is not None:
        if len(t_client.payloads) == 0 or len(t_disp.payloads) == 0:
            print("⚠️ 警告：有一张表被过滤后数据为空，请检查 CSV 内容或过滤条件 (可能被读取为字符串)。")
        else:
            # 运行 Q3
            run_q3_2way_join(
                t_client, t_disp, 
                D=DOMAIN_SIZE, 
                eps_total=1.5, 
                delta_total=1e-4
            )
    else:
        print("❌ 数据加载失败，请检查 CSV 路径。")