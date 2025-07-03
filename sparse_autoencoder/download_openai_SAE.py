import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

with bf.BlobFile(sparse_autoencoder.paths.v5_32k('resid_post_mlp', 5), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    print('downloaded')