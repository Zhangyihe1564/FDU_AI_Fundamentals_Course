import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# 导入 sklearn 组件
from sklearn.metrics import accuracy_score

# 导入各种模型
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# 导入 transformers 和 datasets 组件
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import numpy as np


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# 提取 [CLS] 隐藏层特征的函数（用于 map）
def extract_hidden_states(batch):
    input_text = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**input_text).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

def compare_models(x_train, x_valid, Y_train, Y_valid, save_path=None):
    results = []
    models = {
        "Dummy (Baseline)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000, n_jobs=-1),
        "Linear SVM": SVC(kernel="linear", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Extra Tree": ExtraTreeClassifier(random_state=42),
    }

    print(f"\n开始对比 {len(models)} 个模型...\n" + "=" * 40)
    for name, model in models.items():
        start_ns = time.perf_counter_ns()
        model.fit(x_train, Y_train)
        y_pred = model.predict(x_valid)
        acc = accuracy_score(Y_valid, y_pred)
        elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9  # 秒，精度到纳秒
        print(f"✅ {name:<20} | 准确率: {acc:.4f} | 耗时: {elapsed_s:.4f}s")
        results.append({"Model": name, "Accuracy": acc, "Time (s)": elapsed_s})

    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    names = df_results["Model"].tolist()
    accuracies = df_results["Accuracy"].tolist()
    times = df_results["Time (s)"].tolist()

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 左轴：Accuracy 柱状（偏左）
    bars1 = ax1.bar(x - width/2, accuracies, width, color='skyblue', label='Accuracy')
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)

    # 右轴：Time 柱状（偏右）
    bars2 = ax2.bar(x + width/2, times, width, color='orange', label='Time (s)')
    ax2.set_ylabel("Time (s)")
    max_time = max(times) if times else 1.0
    ax2.set_ylim(0, max_time * 1.3)

    # 在柱子上标注数值
    for xi, h in zip(x - width/2, accuracies):
        ax1.text(xi, h + 0.02, f'{h:.4f}', ha='center', va='bottom', color='black', fontsize=9)
    for xi, t in zip(x + width/2, times):
        ax2.text(xi, t + (max_time * 0.03), f'{t:.4f}s', ha='center', va='bottom', color='orange', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')

    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.title("Classifier Comparison (Accuracy & Time)")
    plt.tight_layout()

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存图片到: ` {save_path} `")

    plt.show()

if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    emotions = load_dataset("emotion")

    # 初始化 tokenizer 和 model（必须在 tokenize/extract_hidden_states 之前）
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    # 对数据集做分词（batched）
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    # 设置输出格式为 torch 张量（使得后续 map 中能将 batch 的 tensors 移动到 device）
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 对编码后的数据集做隐藏层特征抽取（batched），结果包含 "hidden_state"
    emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

    # 从 emotions_hidden 中取出 train/validation 的 hidden_state 和 label 并转为 numpy（得到 X_train, X_valid, y_train, y_valid）
    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])

    plot_save_path = "results/classifier_comparison.png"
    compare_models(X_train, X_valid, y_train, y_valid, save_path=plot_save_path)