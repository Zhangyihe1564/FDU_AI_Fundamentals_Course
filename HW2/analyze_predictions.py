# python
import argparse
import os
import itertools
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

def analyze_single_run(checkpoint, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            if args.fp16 and torch.cuda.is_available():
                model.half()
            model.to(device)
        else:
            tokenizer, model, _ = init_model_and_tokenizer(
                model_ckpt=args.model_ckpt,
                num_labels=args.num_labels,
                cuda_device=args.cuda_device,
                fp16=args.fp16,
                low_cpu_mem_usage=True,
                force_download=False
            )
            model.to(device)
    except Exception as e:
        return None, f"load_error: {e}"

    # prepare datasets
    try:
        emotions, emotions_encoded, _ = prepare_datasets(
            tokenizer=tokenizer,
            model=model,
            device=device,
            dataset_name=args.dataset_name,
            extract_hidden=False,
            max_train_examples=args.max_train_examples,
            max_val_examples=args.max_val_examples,
            max_test_examples=args.max_test_examples
        )
    except Exception as e:
        return None, f"prep_error: {e}"

    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    ds = emotions_encoded[args.split]
    ds = ds.map(lambda batch: forward_pass_with_label(batch, model, tokenizer, device),
                batched=True, batch_size=args.map_batch_size)

    ds.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df = ds[:][cols].copy()

    mean_loss = float(df["loss"].mean())
    accuracy = float((df["label"] == df["predicted_label"]).mean())
    f1 = None
    if f1_score is not None:
        try:
            f1 = float(f1_score(df["label"], df["predicted_label"], average="macro"))
        except Exception:
            f1 = None

    return {"df": df, "mean_loss": mean_loss, "accuracy": accuracy, "f1": f1}, None

def run_sweep(args):
    lr_list = [float(x) for x in args.learning_rates.split(",")] if args.learning_rates else [args.default_lr]
    bs_list = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else [args.default_bs]
    wd_list = [float(x) for x in args.weight_decays.split(",")] if args.weight_decays else [args.default_wd]
    ne_list = [int(x) for x in args.num_epochs.split(",")] if args.num_epochs else [args.default_epochs]

    records = []
    out_root = args.sweep_root
    os.makedirs(out_root, exist_ok=True)

    for lr, bs, wd, ne in itertools.product(lr_list, bs_list, wd_list, ne_list):
        lr_str = safe_val_str(lr)
        wd_str = safe_val_str(wd)
        run_dir = os.path.join(out_root, f"run_lr{lr_str}_bs{bs}_wd{wd_str}_ep{ne}")
        checkpoint = args.checkpoint if args.checkpoint else run_dir
        print("Analyzing:", checkpoint)
        result, error = analyze_single_run(checkpoint, args)
        rec = {
            "checkpoint": checkpoint,
            "learning_rate": lr,
            "batch_size": bs,
            "weight_decay": wd,
            "num_epochs": ne,
            "mean_loss": None,
            "accuracy": None,
            "f1": None,
            "error": error
        }
        if error is None and result:
            df = result["df"]
            # save per-run full csv
            run_out = os.path.join(args.output_dir, os.path.basename(run_dir))
            os.makedirs(run_out, exist_ok=True)
            full_csv = os.path.join(run_out, "analysis_full.csv")
            df.to_csv(full_csv, index=False)
            # save top/bottom
            df.sort_values("loss", inplace=True)
            df.head(args.top_n).to_csv(os.path.join(run_out, f"top_{args.top_n}_smallest_loss.csv"), index=False)
            df.tail(args.top_n).to_csv(os.path.join(run_out, f"top_{args.top_n}_largest_loss.csv"), index=False)

            rec.update({
                "mean_loss": result["mean_loss"],
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "error": None
            })
            print(f"Saved run results to {run_out}")
        else:
            print(f"Run skipped/error: {error}")

        records.append(rec)
        pd.DataFrame(records).to_csv(os.path.join(out_root, "sweep_analysis_summary.csv"), index=False)

    print("Sweep analysis finished. Summary saved to", out_root)

def run_single(args):
    checkpoint = args.checkpoint if args.checkpoint else ""
    result, error = analyze_single_run(checkpoint, args)
    if error:
        print("Error:", error)
        return
    df = result["df"]
    os.makedirs(args.output_dir, exist_ok=True)
    full_csv = os.path.join(args.output_dir, "analysis_full.csv")
    df.to_csv(full_csv, index=False)
    df.sort_values("loss", inplace=True)
    df.head(args.top_n).to_csv(os.path.join(args.output_dir, f"top_{args.top_n}_smallest_loss.csv"), index=False)
    df.tail(args.top_n).to_csv(os.path.join(args.output_dir, f"top_{args.top_n}_largest_loss.csv"), index=False)
    summary = {
        "mean_loss": result["mean_loss"],
        "accuracy": result["accuracy"],
        "f1": result["f1"]
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "analysis_summary.csv"), index=False)
    compute_and_save_confusion_matrix(df, args.output_dir, prefix="confusion_matrix", normalize=False)
    compute_and_save_confusion_matrix(df, args.output_dir, prefix="confusion_matrix_normalized", normalize=True)
    print("Saved analysis to", args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Analyze model predictions by loss; supports sweep mode.")
    parser.add_argument("--checkpoint", type=str, default="", help="path to fine-tuned checkpoint (takes precedence)")
    parser.add_argument("--model_ckpt", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="emotion")
    parser.add_argument("--num_labels", type=int, default=6)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--map_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    # sweep options
    parser.add_argument("--sweep", action="store_true", help="enable sweep mode")
    parser.add_argument("--sweep_root", type=str, default="sweep_results", help="root where sweep runs are stored")
    parser.add_argument("--learning_rates", type=str, default="", help="comma-separated learning rates for sweep")
    parser.add_argument("--batch_sizes", type=str, default="", help="comma-separated batch sizes for sweep")
    parser.add_argument("--weight_decays", type=str, default="", help="comma-separated weight decays for sweep")
    parser.add_argument("--num_epochs", type=str, default="", help="comma-separated num epochs for sweep")
    # defaults if not passing lists
    parser.add_argument("--default_lr", type=float, default=1e-3)
    parser.add_argument("--default_bs", type=int, default=8)
    parser.add_argument("--default_wd", type=float, default=0.0)
    parser.add_argument("--default_epochs", type=int, default=20)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


def compute_and_save_confusion_matrix(df, out_dir, prefix="confusion_matrix", normalize=False):
    """
    df: pandas.DataFrame，必须包含 'label' 和 'predicted_label' 列（整数）
    out_dir: 输出目录（已存在或调用前确保存在）
    prefix: 输出文件前缀
    normalize: 是否按行归一化（True/False）
    """
    if confusion_matrix is None or ConfusionMatrixDisplay is None:
        print("sklearn not available: skipping confusion matrix generation")
        return

    y_true = df["label"].astype(int).values
    y_pred = df["predicted_label"].astype(int).values
    # 使用 labels 保证行列一致并按升序显示
    labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    if len(labels) == 0:
        print("empty labels: skipping confusion matrix")
        return

    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)

    # 保存 CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    cm_df.to_csv(csv_path, index=True)

    # 绘图并保存 PNG
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

if __name__ == "__main__":
    main()
