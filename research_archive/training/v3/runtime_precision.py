"""Explicit FP32 IEEE policy for this isolated candidate; BF16 AMP unchanged."""
import torch

def snapshot():
    return dict(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision())

def configure():
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    current=snapshot()
    if current!=dict(cudnn_allow_tf32=False,matmul_allow_tf32=False,float32_matmul_precision='highest'):
        raise RuntimeError('Explicit TF32-off FP32 policy could not be established')
    return current
