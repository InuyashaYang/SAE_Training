# sparse_autoencoder/my_train.py

import os
import torch
import torch.nn as nn
import h5py
import wandb
from dataclasses import dataclass
from typing import Iterable, Iterator
from tqdm import tqdm

# 关键修改：使用相对导入，从当前包中导入其他模块
# 假设 FastAutoencoder 等核心逻辑已经包含在本文件中
# 如果需要从其他文件导入，例如 model.py, 可以用 from .model import Autoencoder

# ==============================================================================
# 1. 从 sparse_autoencoder/train.py 复制的核心代码 (已恢复AuxK逻辑)
# ==============================================================================

# --- 并行通信简化 ---
@dataclass
class ShardingComms:
    def sh_allreduce_forward(self, x): return x
    def sh_allreduce_backward(self, x): return x
    def init_broadcast_(self, autoencoder): pass
    def dp_allreduce_(self, autoencoder): pass
    def sh_allreduce_scale(self, scaler): pass
    def sh_sum(self, x): return x
    def all_broadcast(self, x): return x

TRIVIAL_COMMS = ShardingComms()

# --- Autoencoder 模型 (恢复了AuxK逻辑) ---
class FastAutoencoder(nn.Module):
    def __init__(self, n_dirs_local, d_model, k, auxk, dead_steps_threshold, comms):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.comms = comms
        self.dead_steps_threshold = dead_steps_threshold
        
        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))
        
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))
        
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_(self)

    def auxk_mask_fn(self, x):
        # 模拟原始代码中的dead_mask逻辑
        dead_mask = (self.stats_last_nonzero > self.dead_steps_threshold).float()
        x = x * dead_mask # 乘以0或1
        return x

    def forward(self, x):
        x_centered = x - self.pre_bias
        latents_pre_act = nn.functional.linear(x_centered, self.encoder.weight, self.latent_bias)
        
        # TopK for main loss
        vals, inds = torch.topk(latents_pre_act, self.k, dim=-1)
        
        # 更新神经元活跃度统计
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(0, inds.reshape(-1), (vals > 1e-3).to(tmp.dtype).reshape(-1))
        self.stats_last_nonzero *= (1 - tmp.clamp(max=1))
        self.stats_last_nonzero += 1
        
        # TopK for AuxK loss
        auxk_vals, auxk_inds = None, None
        if self.auxk is not None:
            masked_latents = self.auxk_mask_fn(latents_pre_act.clone()) # clone to avoid in-place modification issues
            auxk_vals, auxk_inds = torch.topk(masked_latents, self.auxk, dim=-1)

        latents = torch.relu(vals)
        recons = self.decode_sparse(inds, latents)
        
        info = {
            "auxk_inds": auxk_inds,
            "auxk_vals": torch.relu(auxk_vals) if auxk_vals is not None else None,
        }
        
        return recons + self.pre_bias, info

    def decode_sparse(self, inds, vals):
        recons = torch.zeros(inds.shape[0], self.n_dirs_local, device=inds.device, dtype=vals.dtype)
        recons.scatter_(1, inds, vals)
        return self.decoder(recons)

# ... (此处省略与上一版相同的 unit_norm_decoder_, unit_norm_decoder_grad_adjustment_, Logger, batch_tensors)
def unit_norm_decoder_(autoencoder):
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0, keepdim=True)

def unit_norm_decoder_grad_adjustment_(autoencoder):
    if autoencoder.decoder.weight.grad is None: return
    grad = autoencoder.decoder.weight.grad
    proj = torch.einsum("ij,ij->j", grad, autoencoder.decoder.weight.data)
    autoencoder.decoder.weight.grad -= proj * autoencoder.decoder.weight.data

class Logger:
    def __init__(self, **kws):
        self.vals = {}
        self.enabled = not kws.pop("dummy", False)
        if self.enabled:
            wandb.init(**kws)

    def logkv(self, k, v):
        if self.enabled:
            self.vals[k] = v.detach().item() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}

def training_loop_(ae, train_acts_iter, loss_fn, lr, comms, eps, clip_grad, logger):
    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    
    for i, flat_acts_train_batch in enumerate(tqdm(train_acts_iter, desc="Training")):
        flat_acts_train_batch = flat_acts_train_batch.cuda()
        
        recons, info = ae(flat_acts_train_batch)
        loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)
        
        loss.backward()
        unit_norm_decoder_grad_adjustment_(ae)
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(ae.parameters(), clip_grad)
            
        opt.step()
        opt.zero_grad()
        unit_norm_decoder_(ae)
        
        logger.dumpkvs()

def batch_tensors(it: Iterable[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    buffer = []
    current_size = 0
    for t in it:
        buffer.append(t)
        current_size += t.shape[0]
        while current_size >= batch_size:
            concatenated = torch.cat(buffer, dim=0)
            yield concatenated[:batch_size]
            buffer = [concatenated[batch_size:]]
            current_size -= batch_size
    if buffer and buffer[0].shape[0] > 0:
        yield torch.cat(buffer, dim=0)

def init_from_data_(ae, stats_acts_sample):
    from geom_median.torch import compute_geometric_median
    median = compute_geometric_median(stats_acts_sample.float().cpu()).median.cuda().float()
    ae.pre_bias.data = median


# ==============================================================================
# 2. 我们自己实现的、针对HDF5的数据加载逻辑
# ==============================================================================
# ... (此处省略与上一版相同的 create_h5_act_iterator 和 get_stats_sample)
def create_h5_act_iterator(h5_path: str, d_model: int) -> Iterator[torch.Tensor]:
    print(f"INFO: 开始从 {h5_path} 流式加载激活值...")
    with h5py.File(h5_path, 'r') as f:
        dset = f['hidden_states']
        num_sequences = dset.shape[0]
        
        for i in range(num_sequences):
            seq_block = torch.from_numpy(dset[i, :, :])
            yield seq_block.reshape(-1, d_model)

def get_stats_sample(h5_path: str, num_samples: int) -> torch.Tensor:
    print(f"INFO: 从 {h5_path} 提取 {num_samples} 个样本用于初始化...")
    samples = []
    num_collected = 0
    with h5py.File(h5_path, 'r') as f:
        dset = f['hidden_states']
        for i in range(dset.shape[0]):
            seq_block = torch.from_numpy(dset[i, :, :])
            samples.append(seq_block.reshape(-1, dset.shape[2]))
            num_collected += samples[-1].shape[0]
            if num_collected >= num_samples:
                break
    return torch.cat(samples, dim=0)[:num_samples]

# ==============================================================================
# 3. 主程序：配置并启动训练
# ==============================================================================
@dataclass
class Config:
    # 与原始train.py保持一致
    h5_path: str = "/data0/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/train.h5"
    d_model: int = 768
    n_dirs: int = 32768
    bs: int = 4096 # token-level batch size
    k: int = 32
    auxk: int = 256
    auxk_coef: float = 1 / 32
    lr: float = 1e-4
    eps: float = 1e-8
    clip_grad: float | None = 1.0
    dead_toks_threshold: int = 10_000_000
    wandb_project: str = "sparse-autoencoder-full-logic"
    wandb_name: str | None = "gpt2-h5-resid-post-exp1-full"

def normalized_mse(y_hat, y):
    return (y_hat - y).pow(2).sum() / y.pow(2).sum()

def main():
    cfg = Config()
    comms = TRIVIAL_COMMS

    acts_iter = create_h5_act_iterator(cfg.h5_path, cfg.d_model)
    stats_acts_sample = get_stats_sample(cfg.h5_path, num_samples=65536).cuda()

    ae = FastAutoencoder(
        n_dirs_local=cfg.n_dirs,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
    ).cuda()
    
    init_from_data_(ae, stats_acts_sample)

    mse_scale = 1 / (stats_acts_sample.var(dim=0).mean()).item()
    logger = Logger(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.__dict__)

    # 恢复了AuxK的损失函数
    def loss_fn(ae, flat_acts_train_batch, recons, info, logger):
        # 主MSE损失
        main_mse = (recons - flat_acts_train_batch).pow(2).mean()
        logger.logkv("train_mse_unscaled", main_mse)
        
        # AuxK损失
        auxk_loss = torch.tensor(0.0, device=main_mse.device)
        if info["auxk_inds"] is not None:
            auxk_recons = ae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
            residual = (flat_acts_train_batch - recons).detach()
            auxk_loss = normalized_mse(auxk_recons, residual)
        
        logger.logkv("train_auxk_loss", auxk_loss)
        
        return main_mse * mse_scale + cfg.auxk_coef * auxk_loss

    training_loop_(
        ae,
        batch_tensors(acts_iter, cfg.bs),
        loss_fn,
        lr=cfg.lr,
        eps=cfg.eps,
        clip_grad=cfg.clip_grad,
        logger=logger,
        comms=comms,
    )

if __name__ == "__main__":
    main()
