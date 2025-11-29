import torch, torch.nn as nn
class PatchScanner(nn.Module):
    def __init__(self,patch=32,stride=16):
        super().__init__(); self.patch=patch; self.stride=stride
        self.det=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,2))
    def forward(self,x):
        B,C,H,W=x.shape; patches=[]; coords=[]
        for y in range(0,H-self.patch,self.stride):
            for x0 in range(0,W-self.patch,self.stride):
                patches.append(x[:,:,y:y+self.patch,x0:x0+self.patch]); coords.append((x0,y))
        p=torch.cat(patches,0); out=self.det(p); return out, coords
