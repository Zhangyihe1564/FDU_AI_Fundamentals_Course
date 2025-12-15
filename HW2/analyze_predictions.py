import os
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

from model import init_model_and_tokenizer, prepare_datasets

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except Exception:
    confusion_matrix = None
    ConfusionMatrixDisplay = None

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

# python
# ==================== 手动调整超参数 ====================
CHECKPOINT = r"./finetuned_models/emotion_distilbert_ft"  # 若为空则使用默认模型，否则填写checkpoint路径
MODEL_CKPT = "distilbert-base-uncased"
DATASET_NAME = "emotion"
NUM_LABELS = 6
FP16 = True
SPLIT = "validation"
TOP_N = 10
MAP_BATCH_SIZE = 16
OUTPUT_DIR = "analysis_results"
MAX_TRAIN_EXAMPLES = None
MAX_VAL_EXAMPLES = None
MAX_TEST_EXAMPLES = None
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
# ========================================================


def get_device():
    """自动选择可用的 CUDA 设备，若无则使用 CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("No CUDA device available, using CPU")
        return torch.device("cpu")


def safe_val_str(x):
    s = f"{x}"
    return s.replace(".", "p").replace("-", "m")


def forward_pass_with_label(batch, model, tokenizer, device):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}


def analyze_single_run(checkpoint):
    device = get_device()
    try:
        if checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(device)
        else:
            tokenizer, model, _ = init_model_and_tokenizer(
                model_ckpt=MODEL_CKPT,
                num_labels=NUM_LABELS,
                cuda_device=0,  # 改这里：直接传 0
                fp16=FP16 and torch.cuda.is_available(),
                low_cpu_mem_usage=True,
                force_download=False
            )
    except Exception as e:
        return None, f"load_error: {e}"

    # 后面的代码保持不变
    try:
        emotions, emotions_encoded, _ = prepare_datasets(
            tokenizer=tokenizer,
            model=model,
            device=device,
            dataset_name=DATASET_NAME,
            extract_hidden=False,
            max_train_examples=MAX_TRAIN_EXAMPLES,
            max_val_examples=MAX_VAL_EXAMPLES,
            max_test_examples=MAX_TEST_EXAMPLES
        )
    except Exception as e:
        return None, f"prep_error: {e}"

    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    ds = emotions_encoded[SPLIT]
    ds = ds.map(lambda batch: forward_pass_with_label(batch, model, tokenizer, device),
                batched=True, batch_size=MAP_BATCH_SIZE)

    ds.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df = ds[:][cols].copy()
    for i in range(NUM_LABELS):
        LABEL_MAP.setdefault(i, f"{i}")

    # 强制使用手动映射，不使用自动生成的 LABEL_x 映射
    id2label = {int(k): str(v) for k, v in LABEL_MAP.items()}

    # 将数字标签映射为可读文本列（找不到的用原数字字符串）
    df["label_name"] = df["label"].astype(int).map(id2label).fillna(df["label"].astype(int).astype(str))
    df["predicted_label_name"] = df["predicted_label"].astype(int).map(id2label).fillna(
        df["predicted_label"].astype(int).astype(str))

    print("Using forced id2label mapping:", id2label)

    mean_loss = float(df["loss"].mean())
    accuracy = float((df["label"] == df["predicted_label"]).mean())
    f1 = None
    if f1_score is not None:
        try:
            f1 = float(f1_score(df["label"], df["predicted_label"], average="macro"))
        except Exception:
            f1 = None

    return {"df": df, "mean_loss": mean_loss, "accuracy": accuracy, "f1": f1}, None

def compute_and_save_confusion_matrix(df, out_dir, prefix="confusion_matrix", normalize=False):
    if confusion_matrix is None or ConfusionMatrixDisplay is None:
        print("sklearn not available: skipping confusion matrix generation")
        return

    y_true = df["label"].astype(int).values
    y_pred = df["predicted_label"].astype(int).values
    labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    if len(labels) == 0:
        print("empty labels: skipping confusion matrix")
        return

    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    cm_df.to_csv(csv_path, index=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")
    title = f"{prefix}{' (normalized)' if normalize else ''}"
    ax.set_title(title)
    img_path = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(img_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix CSV: {csv_path}")
    print(f"Saved confusion matrix PNG: {img_path}")


def main():
    result, error = analyze_single_run(CHECKPOINT)
    if error:
        print("Error:", error)
        return

    df = result["df"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_csv = os.path.join(OUTPUT_DIR, "analysis_full.csv")
    df.to_csv(full_csv, index=False)
    df.sort_values("loss", inplace=True)
    df.head(TOP_N).to_csv(os.path.join(OUTPUT_DIR, f"top_{TOP_N}_smallest_loss.csv"), index=False)
    df.tail(TOP_N).to_csv(os.path.join(OUTPUT_DIR, f"top_{TOP_N}_largest_loss.csv"), index=False)

    summary = {
        "mean_loss": result["mean_loss"],
        "accuracy": result["accuracy"],
        "f1": result["f1"]
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "analysis_summary.csv"), index=False)
    compute_and_save_confusion_matrix(df, OUTPUT_DIR, prefix="confusion_matrix", normalize=False)
    compute_and_save_confusion_matrix(df, OUTPUT_DIR, prefix="confusion_matrix_normalized", normalize=True)
    print("Saved analysis to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
