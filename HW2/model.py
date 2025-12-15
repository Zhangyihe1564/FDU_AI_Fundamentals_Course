import os
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from typing import Tuple, Dict, Any

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def init_model_and_tokenizer(
    model_ckpt: str = "distilbert-base-uncased",
    num_labels: int = 6,
    cuda_device: int = 0,
    fp16: bool = True,
    low_cpu_mem_usage: bool = True,
    force_download: bool = False
) -> Tuple[Any, Any, torch.device]:
    # 可选：限制可见 GPU（在程序开始前设置最好）
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_device))
    # 释放显存以便安全加载
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True, force_download=force_download)

    load_kwargs = {}
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, **load_kwargs)
    model.to(device)

    return tokenizer, model, device

def prepare_datasets(
    tokenizer,
    model,
    device,
    dataset_name: str = "emotion",
    max_train_examples: int = None,
    max_val_examples: int = None,
    max_test_examples: int = None,
    extract_hidden: bool = True
):
    emotions = load_dataset(dataset_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    emotions_hidden = None
    if extract_hidden:
        def extract_hidden_states(batch):
            # 将需要的输入移动到 device（保持 input_ids/attention_mask 的 dtype）
            input_text = {}
            for k, v in batch.items():
                if k in tokenizer.model_input_names:
                    t = v.to(device)
                    # 不对 long tensors 做半精度转换；只有浮点张量才转 half
                    if t.is_floating_point() and device.type == "cuda" and next(model.parameters()).dtype == torch.float16:
                        t = t.half()
                    input_text[k] = t
            with torch.no_grad():
                out = model(**input_text)
                last_hidden_state = out.last_hidden_state  # (batch, seq_len, hidden)
            # 取第一个 token 向量并搬回 cpu
            return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

        emotions_hidden = emotions_encoded.map(lambda b: extract_hidden_states(b), batched=True, batch_size=None)

    # 可选裁剪用于快速试验
    if max_train_examples:
        emotions_encoded["train"] = emotions_encoded["train"].select(range(min(max_train_examples, len(emotions_encoded["train"]))))
        if emotions_hidden is not None:
            emotions_hidden["train"] = emotions_hidden["train"].select(range(min(max_train_examples, len(emotions_hidden["train"]))))
    if max_val_examples:
        emotions_encoded["validation"] = emotions_encoded["validation"].select(range(min(max_val_examples, len(emotions_encoded["validation"]))))
        if emotions_hidden is not None:
            emotions_hidden["validation"] = emotions_hidden["validation"].select(range(min(max_val_examples, len(emotions_hidden["validation"]))))
    if max_test_examples:
        emotions_encoded["test"] = emotions_encoded["test"].select(range(min(max_test_examples, len(emotions_encoded["test"]))))
        if emotions_hidden is not None:
            emotions_hidden["test"] = emotions_hidden["test"].select(range(min(max_test_examples, len(emotions_hidden["test"]))))

    return emotions, emotions_encoded, emotions_hidden

def make_trainer(model, tokenizer, emotions_encoded, config: Dict[str, Any]):
    import inspect
    from transformers import DataCollatorWithPadding

    train_batch_size = config.get("per_device_train_batch_size", 8)
    eval_batch_size = 8
    gradient_accumulation = config.get("gradient_accumulation_steps", 1)
    logging_steps = max(1, len(emotions_encoded["train"]) // max(1, train_batch_size // gradient_accumulation) // 10)

    data_collator = DataCollatorWithPadding(tokenizer)

    base_args = {
        "output_dir": config.get("output_dir", "./output"),
        "num_train_epochs": config.get("num_train_epochs", 5),
        "learning_rate": config.get("learning_rate", 1e-3),
        "per_device_train_batch_size": train_batch_size,  # 改这里
        "per_device_eval_batch_size": eval_batch_size,  # 改这里
        "weight_decay": config.get("weight_decay", 0.01),
        "evaluation_strategy": config.get("evaluation_strategy", "epoch"),
        "disable_tqdm": config.get("disable_tqdm", True),
        "logging_steps": logging_steps,
        "push_to_hub": False,
        "log_level": "error",
        "fp16": config.get("fp16", True),
        "gradient_accumulation_steps": gradient_accumulation
    }

    sig = inspect.signature(TrainingArguments.__init__)
    supported_params = set(sig.parameters.keys()) - {"self", "kwargs", "args"}
    filtered_args = {k: v for k, v in base_args.items() if k in supported_params}
    if "evaluation_strategy" in base_args and "evaluation_strategy" not in supported_params:
        filtered_args.setdefault("do_eval", True)

    training_args = TrainingArguments(**filtered_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
        data_collator=data_collator
    )
    return trainer

