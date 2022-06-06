import numpy as np
import torch

x = [torch.tensor([1,2,3], dtype=torch.float32), torch.tensor([1,2,3], dtype=torch.float32), torch.tensor([1,2,3], dtype=torch.float32)]
y = torch.stack(x, dim=1)
z = torch.mean(y, dim=1)
print(z)