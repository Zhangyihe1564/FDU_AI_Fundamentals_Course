import os
import traceback
import pandas as pd
from HW2.model import init_model_and_tokenizer, prepare_datasets, make_trainer
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 配置
learning_rates = [1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4, 8e-4, 1e-3]
per_device_train_batch_sizes = [4, 6, 8, 12, 16, 24, 32]
weight_decays = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
num_epochs = [5, 7, 9, 11, 13, 15, 17, 19]

MAX_RUNS = None
out_dir = "sweep_results"
os.makedirs(out_dir, exist_ok=True)
records = []

common_config = {
    "model_ckpt": "distilbert-base-uncased",
    "num_labels": 6,
    "force_download": False,
    "dataset_name": "emotion",
    "max_train_examples": 4000,
    "max_val_examples": 300,
    "extract_hidden": False,
    "save_emotions_hidden_dir": None
}

# 其他超参数的默认值（当未被扫描时使用）
default_params = {
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "weight_decay": 0.01,
    "num_train_epochs": 5
}

def safe_val_str(x):
    s = f"{x}"
    return s.replace(".", "p").replace("-", "m")

# 定义要单独扫描的轴
sweep_axes = {
    "learning_rate": learning_rates,
    "per_device_train_batch_size": per_device_train_batch_sizes,
    "weight_decay": weight_decays,
    "num_train_epochs": num_epochs
}

total = 0
for axis_name, axis_values in sweep_axes.items():
    for val in axis_values:
        if MAX_RUNS is not None and total >= MAX_RUNS:
            break
        total += 1

        val_str = safe_val_str(val)
        out_subdir = os.path.join(out_dir, f"{axis_name}_{val_str}")

        cfg = common_config.copy()
        # 先填入默认值
        cfg.update(default_params)
        # 再覆盖当前轴的值
        cfg[axis_name] = val
        # 确保 output_dir 在 cfg 中
        cfg["output_dir"] = out_subdir

        print("Running:", cfg["output_dir"])
        error = None
        res = None
        try:
            # 显式加载 tokenizer 与带分类头的模型（分类头会被新初始化）
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                cfg.get("model_ckpt"), use_fast=True, force_download=cfg.get("force_download", False)
            )
            model = DistilBertForSequenceClassification.from_pretrained(
                cfg.get("model_ckpt"),
                num_labels=cfg.get("num_labels", 6),
                low_cpu_mem_usage=cfg.get("low_cpu_mem_usage", True)
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            emotions, emotions_encoded, emotions_hidden = prepare_datasets(
                tokenizer, model, device,
                dataset_name=cfg.get("dataset_name", "emotion"),
                max_train_examples=cfg.get("max_train_examples"),
                max_val_examples=cfg.get("max_val_examples"),
                max_test_examples=cfg.get("max_test_examples"),
                extract_hidden=cfg.get("extract_hidden", True)
            )

            trainer = make_trainer(model, tokenizer, emotions_encoded, cfg)
            trainer.train()
            eval_metrics = trainer.evaluate()
            preds = trainer.predict(emotions_encoded["validation"])
            res = {"eval_metrics": eval_metrics, "preds": getattr(preds, "metrics", preds), "saved_files": {}}

        except Exception as e:
            error = str(e)
            print("Run failed:", error)
            traceback.print_exc()

        eval_accuracy = None
        eval_f1 = None
        if res and isinstance(res, dict):
            em = res.get("eval_metrics") or {}
            if isinstance(em, dict):
                eval_accuracy = em.get("eval_accuracy", em.get("accuracy", em.get("acc")))
                eval_f1 = em.get("eval_f1", em.get("f1"))

        # 记录时把所有四个超参数都写出（以便后续分析）
        rec = {
            "output_dir": cfg["output_dir"],
            "varied_param": axis_name,
            "varied_value": val,
            "learning_rate": cfg.get("learning_rate"),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
            "weight_decay": cfg.get("weight_decay"),
            "num_epochs": cfg.get("num_train_epochs"),
            "eval_accuracy": eval_accuracy,
            "eval_f1": eval_f1,
            "error": error
        }
        records.append(rec)

        df = pd.DataFrame(records)
        csv_path = os.path.join("sweep_results.csv")
        df.to_csv(csv_path, index=False)

print("Sweep finished. Results saved to", out_dir)