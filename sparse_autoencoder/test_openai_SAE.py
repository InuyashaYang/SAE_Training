import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import argparse

# 依赖项
import blobfile as bf
import sparse_autoencoder

# 假设 Autoencoder 类定义在 sparse_autoencoder/models.py 中
from sparse_autoencoder.model import Autoencoder

def iter_h5_batches(h5_path: str, batch_size: int, device: torch.device):
    """
    一个高效的生成器，用于从 HDF5 文件中逐批次流式读取数据。
    它将 (length, 1024, 768) 的数据视为一个巨大的 (length * 1024, 768) 数据池。
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
                yield batch_to_yield.to(dtype=torch.float16, device=device)
                concatenated = concatenated[batch_size:]

            if concatenated.shape[0] > 0:
                buffer = [concatenated]
            else:
                buffer = []
        
        if len(buffer) > 0 and buffer[0].shape[0] > 0:
            final_batch = buffer[0]
            yield final_batch.to(dtype=torch.float16, device=device)
    print("\nINFO: Finished streaming all data.")


def evaluate(model: Autoencoder, data_iter, device: torch.device):
    """
    在给定的数据迭代器上评估模型的重建 MSE。
    """
    model.eval()
    total_mse = 0.0
    total_elements = 0

    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Evaluating MSE"):
            batch = batch.to(device)
            _, _, recons = model(batch)
            
            mse_sum_batch = F.mse_loss(recons, batch, reduction='sum')
            
            total_mse += mse_sum_batch.item()
            total_elements += batch.numel()

    avg_mse = total_mse / total_elements if total_elements > 0 else 0
    return avg_mse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a specific Sparse Autoencoder (resid_post_mlp, index 5) on the Wikitext-103 dataset."
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
        default=4096, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on ('cuda' or 'cpu')."
    )
    args = parser.parse_args()
    
    # --- 硬编码模型参数 ---
    layer_name = "resid_post_mlp"
    layer_index = 5
    # -----------------------

    print("--- SAE Evaluation Script ---")
    print(f"Hardcoded Model: Layer='{layer_name}', Index={layer_index}")
    print(f"Dataset Path: {args.h5_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("-----------------------------")

    device = torch.device(args.device)

    # 1. 加载模型（使用硬编码的参数）
    print("Loading model...")
    try:
        model_path = sparse_autoencoder.paths.v5_32k(layer_name, layer_index)
        print(f"Attempting to download model from: {model_path}")
        
        with bf.BlobFile(model_path, mode="rb") as f:
            state_dict = torch.load(f, map_location=device)
            model = Autoencoder.from_state_dict(state_dict)
        
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'blobfile' and 'sparse_autoencoder' are installed and configured correctly.")
        return

    # 2. 创建数据迭代器
    data_iterator = iter_h5_batches(args.h5_path, args.batch_size, device)

    # 3. 运行评估
    print("Starting evaluation...")
    mean_squared_error = evaluate(model, data_iterator, device)

    # 4. 打印结果
    print("\n--- Evaluation Complete ---")
    print(f"Model: {layer_name}, Index: {layer_index}")
    print(f"Final Mean Squared Error (MSE) on test set: {mean_squared_error:.8f}")
    print("-----------------------------")

if __name__ == "__main__":
    main()
