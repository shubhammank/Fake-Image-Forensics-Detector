import torch, torch.nn as nn
class ReconDetector(nn.Module):
 def __init__(self): super().__init__(); self.e=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU()); self.d=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
 def forward(self,x): r=self.d(self.e(x)); return ((x-r)**2).mean([1,2,3])
