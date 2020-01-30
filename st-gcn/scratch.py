import torch
import torch.nn as nn

A = torch.rand([50,2,30,30])
conv = nn.Conv2d(in_channels=A.shape[2], out_channels=5, kernel_size=5)

B = conv(A)
print(B)
