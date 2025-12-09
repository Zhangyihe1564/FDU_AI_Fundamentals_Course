# python
# File: `HW2/hyperparam_sweep.py`
import os
import itertools
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from HW2.model import init_model_and_tokenizer, prepare_datasets, make_trainer

# 配置网格
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [4, 8, 16]
weight_decays = [0.0, 0.01, 0.1]
num_epochs = [20]

MAX_RUNS = None
out_dir = "sweep_results"
os.makedirs(out_dir, exist_ok=True)
records = []

common_config = {
    "model_ckpt": "distilbert-base-uncased",
    "num_labels": 6,
    "force_download": False,
    "dataset_name": "emotion",
    "max_train_examples": 64,
    "max_val_examples": 64,
    "extract_hidden": False,
    "save_emotions_hidden_dir": None
}

def safe_val_str(x):
    s = f"{x}"
    return s.replace(".", "p").replace("-", "m")

total = 0
for lr, bs, wd, ne in itertools.product(learning_rates, batch_sizes, weight_decays, num_epochs):
    if MAX_RUNS is not None and total >= MAX_RUNS:
        break
    total += 1

    lr_str = safe_val_str(lr)
    wd_str = safe_val_str(wd)
    out_subdir = os.path.join(out_dir, f"run_lr{lr_str}_bs{bs}_wd{wd_str}_ep{ne}")

    cfg = common_config.copy()
    cfg.update({
        "learning_rate": lr,
        "per_device_train_batch_size": bs,
        "weight_decay": wd,
        "num_train_epochs": ne,
        "output_dir": out_subdir
    })

    print("Running:", cfg["output_dir"])
    error = None
    res = None
    try:
        # 1. 初始化模型与分词器
        tokenizer, model, device = init_model_and_tokenizer(
            model_ckpt=cfg.get("model_ckpt"),
            num_labels=cfg.get("num_labels", 6),
            fp16=cfg.get("fp16", True),
            force_download=cfg.get("force_download", False)
        )

        # 2. 准备数据集（可裁剪用于快速试验）
        emotions, emotions_encoded, emotions_hidden = prepare_datasets(
            tokenizer, model, device,
            dataset_name=cfg.get("dataset_name", "emotion"),
            max_train_examples=cfg.get("max_train_examples"),
            max_val_examples=cfg.get("max_val_examples"),
            max_test_examples=cfg.get("max_test_examples"),
            extract_hidden=cfg.get("extract_hidden", True)
        )

        # 3. 构建 Trainer 并训练/评估
        trainer = make_trainer(model, tokenizer, emotions_encoded, cfg)
        trainer.train()
        eval_metrics = trainer.evaluate()
        preds = trainer.predict(emotions_encoded["validation"])
        res = {"eval_metrics": eval_metrics, "preds": getattr(preds, "metrics", preds), "saved_files": {}}

    except Exception as e:
        error = str(e)
        print("Run failed:", error)
        traceback.print_exc()

    # 兼容不同 evaluate 返回键名
    eval_accuracy = None
    eval_f1 = None
    if res and isinstance(res, dict):
        em = res.get("eval_metrics") or {}
        if isinstance(em, dict):
            eval_accuracy = em.get("eval_accuracy", em.get("accuracy", em.get("acc")))
            eval_f1 = em.get("eval_f1", em.get("f1"))

    rec = {
        "output_dir": cfg["output_dir"],
        "learning_rate": lr,
        "batch_size": bs,
        "weight_decay": wd,
        "num_epochs": ne,
        "eval_accuracy": eval_accuracy,
        "eval_f1": eval_f1,
        "error": error
    }
    records.append(rec)

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

# 绘图（同原逻辑）
df = pd.DataFrame(records)
if not df.empty:
    df_plot = df.copy()
    df_plot["eval_accuracy"] = pd.to_numeric(df_plot["eval_accuracy"], errors="coerce")
    if df_plot["eval_accuracy"].notna().any():
        summary = df_plot.groupby("learning_rate").eval_accuracy.mean()
        fig, ax = plt.subplots()
        ax.plot(summary.index.astype(str), summary.values, marker='o')
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("eval_accuracy")
        ax.set_title("Learning rate vs Eval Accuracy (mean over runs)")
        fig_path = os.path.join(out_dir, "lr_vs_accuracy.png")
        fig.savefig(fig_path)
        plt.close(fig)
    else:
        print("No numeric eval_accuracy values found; skipping lr_vs_accuracy plot.")

print("Sweep finished. Results saved to", out_dir)
