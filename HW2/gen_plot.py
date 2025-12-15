import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
df = pd.read_csv("sweep_results.csv")
df.columns = df.columns.str.strip()

df["varied_param"] = df["varied_param"].astype(str)
# 更强的清洗：去除不可见字符、压缩空白、统一大小写
df["varied_param"] = (
    df["varied_param"]
    .str.replace("\u200b", "", regex=False)  # 零宽字符
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
)
# 统一命名：将可能出现的错误或旧名映射到标准名
df["varied_param"] = (
    df["varied_param"]
    .str.replace("um_train_epochs", "num_train_epochs", regex=False)
    .str.replace("num_epoch", "num_train_epochs", regex=False)  # 单复数/拼写差异
    .str.replace("num_epochs", "num_train_epochs", regex=False)
    .str.replace("nnum_epochs", "num_train_epochs", regex=False)
    .str.replace("nnum_train_epochs", "num_train_epochs", regex=False)
)

# 调试输出：查看唯一值与计数，帮助定位问题
print("varied_param 唯一值:", df["varied_param"].unique())
print("varied_param 计数:\n", df["varied_param"].value_counts())

# 创建输出目录
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 定义要绘制的参数（确保名称与修正后一致）
params = ["learning_rate", "per_device_train_batch_size", "weight_decay", "num_train_epochs"]

for param in params:
    # 筛选当前参数的实验
    subset = df[df["varied_param"] == param].copy()
    if subset.empty:
        print(f"警告：未找到参数 {param} 的数据。当前 available: {df['varied_param'].unique()}")
        continue

    # 转换 varied_value 为数值（安全转换）
    subset["varied_value"] = pd.to_numeric(subset["varied_value"], errors="coerce")
    subset = subset.dropna(subset=["varied_value"])
    subset = subset.sort_values("varied_value")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(subset["varied_value"], subset["eval_accuracy"], marker='o', label='Accuracy', linewidth=2)
    plt.plot(subset["varied_value"], subset["eval_f1"], marker='s', label='F1 Score', linewidth=2)

    plt.xlabel(param.replace("_", " ").title())
    plt.ylabel("Score")
    plt.title(f"Performance vs {param.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 启用科学记数法（对小数值友好），但保留 0
    plt.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))

    # 保存
    plt.savefig(os.path.join(output_dir, f"{param}.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ 所有图表已生成并保存到 '{output_dir}' 文件夹。")