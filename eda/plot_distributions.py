import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 核心修改：动态获取目录路径
# ==========================================
# 1. 获取当前脚本所在目录 (eda 文件夹)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 数据源文件夹：从 eda 退回上一级，再进入 financial
DATA_BASE_PATH = os.path.join(SCRIPT_DIR, '../financial/')

# 3. 图片输出文件夹：从 eda 退回上一级，再进入 density_picture
OUTPUT_PIC_DIR = os.path.join(SCRIPT_DIR, '../density_picture/')


def plot_filtered_density(csv_path, filter_attribute, filter_value, target_attribute, bins=50):
    """
    读取 CSV 文件，按照指定条件过滤后，画出目标属性的数据分布图。
    
    参数:
    - csv_path: CSV 文件绝对路径
    - filter_attribute: 用来过滤的属性A (如 'operation')
    - filter_value: 过滤的值 (如 'VYBER KARTOU')
    - target_attribute: 需要观察密集/稀疏的属性B (如 'account_id' 或 'amount')
    - bins: 直方图的柱子数量，默认 50
    """
    # 提前从 csv_path 中提取表名 (例如把 '.../financial/trans.csv' 变成 'trans')
    table_name = os.path.basename(csv_path).replace('.csv', '')
    
    print(f"正在读取并处理 {table_name}.csv ...")
    
    try:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
    except FileNotFoundError:
        print(f"⚠️ 找不到文件: {csv_path}")
        return
    
    # 1. 过滤数据
    df_filtered = df[df[filter_attribute] == filter_value]
    
    if df_filtered.empty:
        print(f"警告：没有找到 {filter_attribute} == '{filter_value}' 的数据！")
        return

    # 2. 提取目标属性列，并丢弃可能存在的空值 (NaN)
    data_to_plot = df_filtered[target_attribute].dropna()

    # ==========================================
    # 新增：计算总数和 Distinct 数
    # ==========================================
    total_count = len(data_to_plot)
    distinct_count = data_to_plot.nunique()
    
    # 3. 开始画图
    plt.figure(figsize=(10, 6))
    
    # 画直方图
    plt.hist(data_to_plot, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加标题（这里加入了表名！）和标签
    plt.title(f"[{table_name.upper()}] Density of '{target_attribute}' (Filtered by {filter_attribute}='{filter_value}')", fontsize=14)
    plt.xlabel(target_attribute, fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)

    # ==========================================
    # 新增：在图片右上角添加数据统计文本框
    # ==========================================
    stats_text = f"Total Count: {total_count}\nDistinct Count: {distinct_count}"
    # 0.95, 0.95 表示在相对于整个图表轴坐标的右上角位置
    plt.gca().text(0.95, 0.95, stats_text, 
                   transform=plt.gca().transAxes, 
                   fontsize=12,
                   verticalalignment='top', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 检查图片输出文件夹是否存在，如果不存在则自动创建
    os.makedirs(OUTPUT_PIC_DIR, exist_ok=True)
    
    # 动态生成包含完整信息的图片文件名
    output_filename = f"{table_name}_filter_{filter_attribute}_{filter_value}_target_{target_attribute}.png".replace(" ", "_")
    
    # 将输出文件夹路径和文件名拼接起来
    full_filepath = os.path.join(OUTPUT_PIC_DIR, output_filename)
    
    # 规范化路径显示
    full_filepath = os.path.normpath(full_filepath)
    
    # 保存图片到指定文件夹
    plt.savefig(full_filepath, dpi=300)
    print(f"✅ 图片绘制完成！已保存到: {full_filepath}")
    
    # 清空画布
    plt.clf()

# --- 运行测试 ---
if __name__ == "__main__":
    print("开始批量生成 Workload SQL 中的 Join Key 频率分布图...\n")

    # 1. Client 表 (来自 Q3)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'client.csv'), 'district_id', 18, 'client_id')
    
    # 2. Disp 表 (来自 Q3)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'disp.csv'), 'type', 'DISPONENT', 'client_id')
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'disp.csv'), 'type', 'DISPONENT', 'account_id')

    # 3. Account 表 (来自 Q4-Q8)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'account.csv'), 'district_id', 18, 'account_id')

    # 4. Trans 表 (来自 Q4-Q8)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'trans.csv'), 'operation', 'VYBER KARTOU', 'account_id')

    # 5. Order 表 (来自 Q5-Q8)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'order.csv'), 'k_symbol', 'LEASING', 'account_id')

    # 6. Loan 表 (来自 Q8)
    plot_filtered_density(os.path.join(DATA_BASE_PATH, 'loan.csv'), 'duration', 36, 'account_id')

    print("\n🎉 所有分布图已生成完毕，请在 density_picture 文件夹中查看！")