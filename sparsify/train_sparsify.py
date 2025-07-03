import os
import torch
import h5py
from datasets import Dataset
from transformers import AutoModelForCausalLM

# 从 sparsify 库导入核心模块
from sparsify import SaeConfig, Trainer, TrainConfig

# ==============================================================================
# 0. 强制本地化设置
# ==============================================================================
os.environ['HF_DATASETS_OFFLINE'] = "1"
print("环境变量 HF_DATASETS_OFFLINE 已设置为 '1'，将强制使用本地缓存。")


# ==============================================================================
# 1. 核心配置
# ==============================================================================
MODEL_NAME = "gpt2"
HOOK_POINTS = ["h.5"] 
TRAIN_H5_PATH = "/data0/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/train.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- 使用 sparsify Trainer 进行 On-the-fly 训练 (最终正确版) ---")
print(f"基础模型: {MODEL_NAME}")
print(f"目标钩子点: {HOOK_POINTS}")
print(f"设备: {DEVICE}")
print("-------------------------------------------------")


# ==============================================================================
# 2. 数据加载函数
# ==============================================================================
def load_h5_to_dataset(h5_path: str) -> Dataset:
    """从HDF5文件加载input_ids并创建datasets.Dataset对象。"""
    print(f"正在从 {h5_path} 手动加载数据...")
    with h5py.File(h5_path, 'r') as f:
        if 'input_ids' not in f:
            raise ValueError(f"文件 {h5_path} 中没有找到 'input_ids' 数据集")
        input_ids = f['input_ids'][:]
        print(f"成功加载数据，形状: {input_ids.shape}")
        return Dataset.from_dict({"input_ids": input_ids})

# ==============================================================================
# 3. 执行加载并设置格式
# ==============================================================================
print("创建训练数据集...")
try:
    train_dataset = load_h5_to_dataset(TRAIN_H5_PATH)
    
    # 关键修正：设置数据集的输出格式为PyTorch张量
    train_dataset.set_format(type="torch", columns=["input_ids"])
    
    print("训练数据集创建成功，并已设置输出格式为PyTorch张量！")
except Exception as e:
    print(f"创建或设置数据集格式时出错: {e}")
    exit()

print(f"\n正在加载基础模型 '{MODEL_NAME}' (强制离线)...")
try:
    gpt_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    print("基础模型加载成功！")
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()


# ==============================================================================
# 4. 配置并运行训练
# ==============================================================================
# a. 创建SaeConfig
sae_config = SaeConfig(
    expansion_factor=48,
    k=32
)

# b. 创建TrainConfig
train_config = TrainConfig(
    sae=sae_config,
    batch_size=32,
    hookpoints=HOOK_POINTS,
    lr_warmup_steps=5000,
    log_to_wandb=True,
    run_name="gpt2-h5-resid-post-exp1", 
    wandb_log_frequency=200,
    save_every=5000,
)

# c. 初始化并启动训练
trainer = Trainer(train_config, train_dataset, gpt_model)
print("\nTrainer配置完成，准备开始训练...")
print("注意：训练将持续进行，请通过W&B监控，并在适当时机手动停止 (Ctrl+C)。")
trainer.fit()

print("\n训练完成！(或被手动停止)")
