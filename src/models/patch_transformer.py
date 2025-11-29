import torch, torch.nn as nn
class PatchTransformer(nn.Module):
    def __init__(self): super().__init__(); self.embed=nn.Conv2d(3,256,32,32); enc=nn.TransformerEncoderLayer(256,4,512); self.trans=nn.TransformerEncoder(enc,4); self.cls=nn.Linear(256,2)
    def forward(self,x): x=self.embed(x); B,C,H,W=x.shape; x=x.flatten(2).permute(2,0,1); x=self.trans(x); return self.cls(x.mean(0))
