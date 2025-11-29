import torch, torch.nn as nn, torchvision.models as m
class CNNBaseline(nn.Module):
 def __init__(self): super().__init__(); self.b=m.efficientnet_b0(weights=None); self.b.classifier[1]=nn.Linear(1280,2)
 def forward(self,x): return self.b(x)
