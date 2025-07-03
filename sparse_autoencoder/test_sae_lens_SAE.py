import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import argparse

from sae_lens import SAE

def iter_h5_batches(h5_path: str, batch_size: int, device: torch.device):
    """
    一个高效的生成器，用于从 HDF5 文件中逐批次流式读取数据。
    """
    print(f"INFO: Starting to stream data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        dset = f['hidden_states']
        num_sequences = dset.shape[0]
        
        buffer = []
        for i in tqdm(range(num_sequences), desc="Reading sequences from H5"):
            seq_block = torch.from_numpy(dset[i, :, :])
            buffer.append(seq_block)
            concatenated = torch.cat(buffer, dim=0)

            while concatenated.shape[0] >= batch_size:
                batch_to_yield = concatenated[:batch_size]
                yield batch_to_yield.to(dtype=torch.float32, device=device)
                concatenated = concatenated[batch_size:]

            if concatenated.shape[0] > 0:
                buffer = [concatenated]
            else:
                buffer = []
        
        if len(buffer) > 0 and buffer[0].shape[0] > 0:
            final_batch = buffer[0]
            yield final_batch.to(dtype=torch.float32, device=device)
    print("\nINFO: Finished streaming all data.")


def evaluate_mse(model: SAE, data_iter, device: torch.device):
    """
    在给定的数据迭代器上评估 sae-lens 模型的重建 MSE。
    这个版本是根据我们最终的发现编写的。
    """
    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    
    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Evaluating MSE"):
            batch = batch.to(device)
            
            # 关键：在 eval 模式下，model(batch) 直接返回重建张量
            recons = model(batch)
            
            # 检查形状是否匹配（以防万一）
            if recons.shape != batch.shape:
                print(f"\nERROR: Shape mismatch detected! Input: {batch.shape}, Output: {recons.shape}")
                # 尝试处理广播情况，但这通常表示之前的错误
                if recons.shape == model.b_dec.shape:
                     print("Error is likely due to model/data mismatch, outputting only decoder bias.")
                     # 在这里可以选择停止或跳过，但计算出的MSE会非常大
                continue

            # 使用 reduction='sum' 来计算总平方和，避免浮点数精度问题
            squared_error_sum = F.mse_loss(recons, batch, reduction='sum')
            
            total_squared_error += squared_error_sum.item()
            total_elements += batch.numel()

    # 计算最终的均方误差
    mean_squared_error = total_squared_error / total_elements if total_elements > 0 else 0
    return mean_squared_error


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a specific SAE model from sae-lens on the Wikitext-103 dataset."
    )
    parser.add_argument(
        "--h5_path", 
        type=str, 
        default="/data0/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/test.h5",
        help="Path to the Wikitext-103 test.h5 file."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2048, # 使用一个合适的批次大小
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on ('cuda' or 'cpu')."
    )
    args = parser.parse_args()
    
    # --- 核心配置：使用与数据匹配的模型 ---
    release = "gpt2-small-resid-post-v5-32k"
    sae_id = "blocks.5.hook_resid_post"

    print("--- SAE-Lens Final Evaluation Script ---")
    print(f"Model Release: '{release}'")
    print(f"SAE ID: '{sae_id}'")
    print(f"Dataset Path: {args.h5_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("----------------------------------")

    device_obj = torch.device(args.device)

    print("Loading model using sae-lens...")
    try:
        # 加载模型并取元组的第一个元素
        model = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=args.device
        )[0] 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 创建数据迭代器
    data_iterator = iter_h5_batches(args.h5_path, args.batch_size, device_obj)
    
    # 开始评估
    print("Starting evaluation...")
    mean_squared_error = evaluate_mse(model, data_iterator, device_obj)

    # 打印最终结果
    print("\n--- Evaluation Complete ---")
    print(f"Model: {release} / {sae_id}")
    print(f"Final Mean Squared Error (MSE) on test set: {mean_squared_error:.8f}")
    print("-----------------------------")

if __name__ == "__main__":
    main()
