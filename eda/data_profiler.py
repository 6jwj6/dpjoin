import pandas as pd
import os

# ==========================================
# 核心修改：动态获取路径
# ==========================================
# 1. 获取当前脚本 (data_profiler.py) 所在的绝对目录 (即 eda 文件夹)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 从 eda 文件夹退回上一级 (..)，再进入 financial 文件夹
DEFAULT_BASE_PATH = os.path.join(SCRIPT_DIR, '../financial/')


def profile_database(base_path=DEFAULT_BASE_PATH):
    """
    遍历指定目录下的数据库CSV文件，输出每个字段的数据范围和统计特征，
    并在最后输出整个数据库的总体属性汇总。
    """
    tables = ['account.csv', 'trans.csv', 'order.csv', 'client.csv', 'disp.csv', 'loan.csv']
    
    print("开始进行数据库探查 (Data Profiling)...\n")
    
    # 新增：用于存储每张表的汇总信息的列表
    summary_data = []
    
    for table_file in tables:
        file_path = os.path.join(base_path, table_file)
        
        # 将相对路径转换为整洁的绝对路径，防止因为操作系统差异报错
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}，已跳过。")
            continue
            
        print(f"\n{'='*50}")
        print(f"📊 表名: {table_file.upper()}")
        print(f"{'='*50}")
        
        df = pd.read_csv(file_path, sep=';', low_memory=False)
        
        # 获取当前表的总体信息
        total_rows = len(df)
        total_cols = len(df.columns)
        numeric_cols_count = 0
        other_cols_count = 0
        
        for column in df.columns:
            distinct_count = df[column].nunique()
            
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_cols_count += 1
                if df[column].notna().any():
                    min_val = df[column].min()
                    max_val = df[column].max()
                    print(f"🔸 字段: [{column}] (数值型)")
                    print(f"    -> 最小值: {min_val} | 最大值: {max_val} | 总个数: {total_rows} | Distinct个数: {distinct_count}")
                else:
                    print(f"🔸 字段: [{column}] (数值型 - 全为空)")
            else:
                other_cols_count += 1
                print(f"🔹 字段: [{column}] (字符/其他)")
                print(f"    -> 总个数: {total_rows} | Distinct个数: {distinct_count}")
        
        # 新增：将该表的统计结果以字典形式保存到汇总列表中
        summary_data.append({
            '表名 (Table)': table_file.replace('.csv', ''),
            '总行数 (Rows)': total_rows,
            '总列数 (Columns)': total_cols,
            '数值型字段数': numeric_cols_count,
            '字符型字段数': other_cols_count
        })

    # ==========================================
    # 打印最终的全局汇总信息
    # ==========================================
    if summary_data:
        print("\n\n" + "🌟"*25)
        print("    数据库总体属性汇总 (Database Summary)")
        print("🌟"*25)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print("="*55 + "\n")


def find_min_max_locations(csv_path, column_name):
    """
    寻找 CSV 文件中指定列的最大值和最小值，并输出它们所在的行号。
    """
    print(f"\n🔍 开始搜索文件: {os.path.basename(csv_path)} | 目标字段: [{column_name}]")
    
    csv_path = os.path.normpath(csv_path)
    
    try:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
    except FileNotFoundError:
        print(f"⚠️ 找不到文件: {csv_path}")
        return

    # 检查列是否存在
    if column_name not in df.columns:
        print(f"⚠️ 文件中不存在名为 '{column_name}' 的列！")
        return
        
    # 检查是否为数值型数据
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"⚠️ 字段 '{column_name}' 不是数值型数据，无法计算最大/最小值。")
        return

    # 1. 找到最小值和最大值
    min_val = df[column_name].min()
    max_val = df[column_name].max()

    # 2. 找到这些值对应的所有行索引 (Index)
    min_indices = df[df[column_name] == min_val].index.tolist()
    max_indices = df[df[column_name] == max_val].index.tolist()

    # 3. 计算实际的 CSV 物理行号 (Pandas index + 2)
    min_csv_lines = [i + 2 for i in min_indices]
    max_csv_lines = [i + 2 for i in max_indices]

    # 4. 打印结果
    print("-" * 50)
    print(f"🔽 最小值: {min_val}")
    print(f"   共出现了 {len(min_indices)} 次。")
    print(f"   Pandas DataFrame 索引号: {min_indices[:5]}{' ...' if len(min_indices)>5 else ''}")
    print(f"   CSV 实际物理文本行号:  {min_csv_lines[:5]}{' ...' if len(min_csv_lines)>5 else ''}")
    print()
    print(f"🔼 最大值: {max_val}")
    print(f"   共出现了 {len(max_indices)} 次。")
    print(f"   Pandas DataFrame 索引号: {max_indices[:5]}{' ...' if len(max_indices)>5 else ''}")
    print(f"   CSV 实际物理文本行号:  {max_csv_lines[:5]}{' ...' if len(max_csv_lines)>5 else ''}")
    print("-" * 50)

if __name__ == "__main__":
    # 调用汇总探查
    profile_database(base_path=DEFAULT_BASE_PATH)

    # 如果你要测试极值查找，可以使用下面这种拼接路径的方式：
    find_min_max_locations(os.path.join(DEFAULT_BASE_PATH, 'account.csv'), 'account_id')
    # find_min_max_locations(os.path.join(DEFAULT_BASE_PATH, 'trans.csv'), 'amount')