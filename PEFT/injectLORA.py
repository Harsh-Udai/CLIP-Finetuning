from peft.tuners.lora import LoraLayer
from peft.utils.other import transpose
import torch.nn as nn
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.randn(r, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(original_linear.out_features, r) * 0.01)
        self.trainable = True  # helpful for filtering

    def forward(self, x):
        result = self.original(x)
        lora_out = self.dropout(x) @ self.lora_A.T
        lora_out = lora_out @ self.lora_B.T
        return result + self.scale * lora_out

def inject_lora(model, target_modules=("q_proj", "v_proj", "k_proj", "out_proj"), r=32, alpha=64, dropout=0.1):
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_module = get_parent(model, name)
                attr_name = name.split(".")[-1]
                original_linear = getattr(parent_module, attr_name)
                setattr(parent_module, attr_name, LoRALinear(original_linear, r, alpha, dropout))



def get_parent(model, module_name):
    """Helper to get parent module by full name"""
    parts = module_name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    return model