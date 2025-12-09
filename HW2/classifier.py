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

#数据生成
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
    # ==========================================
    # 1. 定义要比较的模型字典
    # ==========================================
    models = {
        "Dummy (Baseline)": DummyClassifier(strategy="most_frequent"),

        # max_iter 设置大一点，因为高维数据通常收敛较慢
        "Logistic Regression": LogisticRegression(max_iter=3000, n_jobs=-1),

        # 线性核 SVM 通常在文本向量上表现很好且比 RBF 核快
        "Linear SVM": SVC(kernel="linear", C=1.0),

        # 随机森林：通用强力模型
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),

        # 梯度提升树 (类似 LightGBM/XGBoost 的 sklearn 原生实现)
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),

        # K-近邻
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),

        # 决策树
        "Decision Tree": DecisionTreeClassifier(random_state=42),

        # 极端随机树
        "Extra Tree": ExtraTreeClassifier(random_state=42),

    }

    # ==========================================
    # 2. 循环训练与评估
    # ==========================================
    results = []

    print(f"\n开始对比 {len(models)} 个模型...\n" + "=" * 40)

    for name, model in models.items():
        start_time = time.time()

        # 训练
        model.fit(x_train, Y_train)

        # 预测
        y_pred = model.predict(x_valid)

        # 评估
        acc = accuracy_score(Y_valid, y_pred)
        elapsed = time.time() - start_time

        print(f"✅ {name:<20} | 准确率: {acc:.4f} | 耗时: {elapsed:.2f}s")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Time (s)": elapsed
        })

    # ==========================================
    # 3. 可视化结果
    # ==========================================
    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 6))

    # 绘制条形图
    bars = plt.barh(df_results["Model"], df_results["Accuracy"], color='skyblue')
    plt.xlabel("Accuracy Score")
    plt.title("Classifier Comparison on Embeddings")
    plt.xlim(0, 1.05)  # 设置x轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 在柱子上显示具体数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', va='center', color='black')

        # 如果提供保存路径，则创建目录并保存（高分辨率）
    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存图片到: `{save_path}`")
    plt.tight_layout()
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