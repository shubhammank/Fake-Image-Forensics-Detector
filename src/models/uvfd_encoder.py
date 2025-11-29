import torch, torch.nn as nn, torch.fft
class UVFDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_conv=nn.Sequential(
            nn.Conv2d(2,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU()
        )
        self.cls=nn.Sequential(nn.Linear(128*56*56,256),nn.ReLU(),nn.Linear(256,2))
    def forward(self,x):
        fft=torch.fft.fft2(x); mag=torch.abs(fft).mean(1); phase=torch.angle(fft).mean(1)
        freq=torch.stack([mag,phase],1); f=self.freq_conv(freq); return self.cls(f.flatten(1))
