import time
from utils import generate_2way_tables, process_single_table, print_join_summary
from join_mechanism import DPJoiner, JoinMetadata

def run_2way_join(table_A, table_B, D, eps_total=1.0, delta_total=1e-6):
    print(f"\n{'='*70}\n 启动 2-Way Join\n{'='*70}")
    t_start = time.time()
    
    eps_meta = eps_total * 0.5
    delta_meta = delta_total * 0.5
    eps_stage = (eps_total - eps_meta) / 3.0 # 2表预处理 + 1次Join
    delta_stage = (delta_total - delta_meta) / 3.0

    meta_A = JoinMetadata.from_base_table([r['Key'] for r in table_A.payloads], eps_meta, delta_meta)
    meta_B = JoinMetadata.from_base_table([r['Key'] for r in table_B.payloads], eps_meta, delta_meta)

    parts_A, bucks_A = process_single_table("Table_A", table_A.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_A.b)
    parts_B, bucks_B = process_single_table("Table_B", table_B.payloads, 'Key', D, eps_stage, delta_stage, eps_stage, delta_stage, meta_B.b)

    meta_AB = meta_A.join(meta_B)
    joiner = DPJoiner(eps_stage, delta_stage, meta_AB.b)
    
    parts_AB, bucks_AB = joiner.run_join(parts_A, bucks_A, parts_B, bucks_B, merge_factor=2)
    print_join_summary(parts_AB, bucks_AB, time.time() - t_start)

if __name__ == "__main__":
    tA, tB = generate_2way_tables(1000000, 100000, 80000)
    run_2way_join(tA, tB, 1000000)