# python
import os
import json
from transformers import pipeline
import torch

OUT_PATH = "adversarial_results.txt"
MODEL_DIR = "./finetuned_models/emotion_distilbert_ft"

adversarial_samples = [
    {
        "text": "Great, another meeting canceled. Couldn't be more thrilled 🙄",
        "intended": "anger",
        "reason": "明显的讽刺（sarcasm）与反语表情，会让模型把文本字面情绪识别为正面。"
    },
    {
        "text": "I laughed until I cried — best and worst night of my life.",
        "intended": "sadness",
        "reason": "混合情绪描述（同时包含 laugh/cry），模型倾向于抓住强正面词汇如 'laughed'。"
    },
    {
        "text": "I can't believe it! That surprise party actually made me so emotional.",
        "intended": "surprise",
        "reason": "包含明显正面词汇（emotional / made me happy）可能导致模型判为 joy。"
    },
    {
        "text": "Wow. 😢",
        "intended": "sadness",
        "reason": "文本极短，语境不足；emoji 可能被标记器当作噪声或与别的标签混淆。"
    },
    {
        "text": "I'm so proud and terrified at the same time.",
        "intended": "fear",
        "reason": "同句包含相互冲突的情绪（proud vs terrified），单标签模型易被正面词覆盖。"
    }
]

def load_classifier(model_dir):
    device = 0 if torch.cuda.is_available() else -1
    try:
        clf = pipeline("text-classification", model=model_dir, return_all_scores=True, device=device)
    except Exception as e:
        raise RuntimeError(f"无法加载模型于 {model_dir}: {e}")
    return clf

def normalize_pred_label(pred_label, clf):
    lab = pred_label.lower()
    # 如果返回是形式 LABEL_X，尝试用模型的 id2label 映射
    if lab.startswith("label_"):
        try:
            idx = int(lab.split("_")[-1])
            id2label = getattr(clf.model.config, "id2label", None)
            if id2label and idx in id2label:
                return id2label[idx].lower()
        except Exception:
            pass
    return lab

def run_and_save(clf, samples, out_path):
    lines = []
    for i, s in enumerate(samples, 1):
        preds = clf(s["text"], return_all_scores=True)
        # pipeline 返回一个 list（batch）; 单条文本取第0个元素
        scores = preds[0]
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = scores_sorted[0]
        pred_label = normalize_pred_label(top["label"], clf)
        top_score = top["score"]
        # build readable scores
        score_map = {normalize_pred_label(p["label"], clf): p["score"] for p in scores_sorted}
        fooled = pred_label != s["intended"].lower()
        entry = {
            "index": i,
            "text": s["text"],
            "intended": s["intended"],
            "predicted": pred_label,
            "predicted_score": float(top_score),
            "all_scores": {k: float(v) for k, v in score_map.items()},
            "fooled": bool(fooled),
            "explain": s["reason"]
        }
        lines.append(entry)
        # console short print
        print(f"[{i}] intended={s['intended']} predicted={pred_label} fooled={fooled}")
    # 写文件（覆盖）
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 `{out_path}`")

def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"模型目录未找到： `{MODEL_DIR}`")
    clf = load_classifier(MODEL_DIR)
    run_and_save(clf, adversarial_samples, OUT_PATH)

if __name__ == "__main__":
    main()
