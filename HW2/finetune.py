import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from model import init_model_and_tokenizer, prepare_datasets, make_trainer

class ModelFinetune:
    """模型微调和保存管理类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.output_dir = Path(config.get("output_dir", "./model"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_model(self) -> bool:
        """初始化模型和分词器"""
        try:
            self.tokenizer, self.model, self.device = init_model_and_tokenizer(
                model_ckpt=self.config.get("model_ckpt", "distilbert-base-uncased"),
                num_labels=self.config.get("num_labels", 6),
                cuda_device=self.config.get("cuda_device", 0),
                fp16=self.config.get("fp16", True),
                low_cpu_mem_usage=self.config.get("low_cpu_mem_usage", True),
                force_download=self.config.get("force_download", False)
            )
            print(f"✓ 模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False

    def prepare_data(self) -> Optional[Dict]:
        """准备数据集"""
        try:
            emotions, emotions_encoded, emotions_hidden = prepare_datasets(
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.device,
                dataset_name=self.config.get("dataset_name", "emotion"),
                max_train_examples=self.config.get("max_train_examples"),
                max_val_examples=self.config.get("max_val_examples"),
                max_test_examples=self.config.get("max_test_examples"),
                extract_hidden=self.config.get("extract_hidden", False)
            )
            print(f"✓ 数据准备完成")
            print(f"  训练集大小: {len(emotions_encoded['train'])}")
            print(f"  验证集大小: {len(emotions_encoded['validation'])}")
            print(f"  测试集大小: {len(emotions_encoded['test'])}")

            return {
                "emotions": emotions,
                "emotions_encoded": emotions_encoded,
                "emotions_hidden": emotions_hidden
            }
        except Exception as e:
            print(f"✗ 数据准备失败: {e}")
            return None

    def finetune(self, emotions_encoded) -> bool:
        """执行模型微调"""
        try:
            self.trainer = make_trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                emotions_encoded=emotions_encoded,
                config=self.config
            )
            print("✓ 训练器创建成功，开始训练...")
            train_result = self.trainer.train()
            print(f"✓ 训练完成")
            print(f"  最终训练损失: {train_result.training_loss:.4f}")
            return True
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            return False

    def evaluate(self) -> Optional[Dict[str, float]]:
        """评估模型"""
        if self.trainer is None:
            print("✗ 请先进行微调")
            return None

        try:
            eval_result = self.trainer.evaluate()
            print("✓ 评估完成")
            print(f"  准确率: {eval_result.get('eval_accuracy', 0):.4f}")
            print(f"  F1 分数: {eval_result.get('eval_f1', 0):.4f}")
            print(f"  评估损失: {eval_result.get('eval_loss', 0):.4f}")
            return eval_result
        except Exception as e:
            print(f"✗ 评估失败: {e}")
            return None

    def save_model(self, run_name: Optional[str] = None) -> Optional[str]:
        """保存微调后的模型"""
        if self.model is None or self.tokenizer is None:
            print("✗ 模型未初始化")
            return None

        try:
            # 生成时间戳和运行名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = run_name or f"finetuned_{timestamp}"
            save_path = self.output_dir / save_name
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存模型和分词器
            self.model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

            # 保存配置
            config_path = save_path / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            print(f"✓ 模型已保存到: {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"✗ 模型保存失败: {e}")
            return None

    def run(self, run_name: Optional[str] = None) -> Dict[str, Any]:
        """完整的微调流程"""
        result = {
            "success": False,
            "model_path": None,
            "eval_metrics": None,
            "error": None
        }

        # 1. 初始化模型
        if not self.setup_model():
            result["error"] = "模型初始化失败"
            return result

        # 2. 准备数据
        data = self.prepare_data()
        if data is None:
            result["error"] = "数据准备失败"
            return result

        # 3. 执行微调
        if not self.finetune(data["emotions_encoded"]):
            result["error"] = "模型微调失败"
            return result

        # 4. 评估模型
        eval_result = self.evaluate()
        if eval_result is None:
            result["error"] = "模型评估失败"
            return result

        # 5. 保存模型
        model_path = self.save_model(run_name)
        if model_path is None:
            result["error"] = "模型保存失败"
            return result

        result["success"] = True
        result["model_path"] = model_path
        result["eval_metrics"] = eval_result
        return result


def main():
    """示例使用"""
    config = {
        "model_ckpt": "distilbert-base-uncased", #模型基底
        "num_labels": 6, #情感分类标签数
        "dataset_name": "emotion",
        "output_dir": "./finetuned_models",
        "cuda_device": 0,
        "fp16": True,
        "low_cpu_mem_usage": True,
        "per_device_train_batch_size": 16, #训练批次大小
        "num_train_epochs": 5, #训练轮数
        "learning_rate": 5e-5, #学习率
        "weight_decay": 0.001, #权重衰减
        "gradient_accumulation_steps": 1, #梯度累积步数
        "evaluation_strategy": "epoch", #评估策略
        "disable_tqdm": False,
        "max_train_examples": None,
        "max_val_examples": None,
        "max_test_examples": None,
        "extract_hidden": False
    }

    finetune = ModelFinetune(config)
    result = finetune.run(run_name="emotion_distilbert_ft")

    if result["success"]:
        print("\n✓ 微调成功！")
        print(f"模型路径: {result['model_path']}")
        print(f"评估指标: {result['eval_metrics']}")
    else:
        print(f"\n✗ 微调失败: {result['error']}")


if __name__ == "__main__":
    main()

'''
RESULT1
"per_device_train_batch_size": 16, #训练批次大小
"num_train_epochs": 5, #训练轮数
"learning_rate": 8e-5, #学习率
"weight_decay": 0.001, #权重衰减

'eval_loss': 0.3076220452785492, 
'eval_accuracy': 0.936, 
'eval_f1': 0.9359976158517335, 
'eval_runtime': 1.5003, 
'eval_samples_per_second': 1333.09, 
'eval_steps_per_second': 166.636, 
'epoch': 5.0
'''
