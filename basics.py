import torch
import numpy as np

shape = (2,3)
np_array = np.zeros(shape)
torch_tensor = torch.zeros(shape)

# print(np_array)
print(torch_tensor)
print(torch_tensor.shape)
print(torch_tensor.device)

if torch.accelerator.is_available():
    tensor = torch_tensor.to(torch.accelerator.current_accelerator())
    print(tensor.device)