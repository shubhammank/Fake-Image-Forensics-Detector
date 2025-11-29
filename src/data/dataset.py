import os
from torch.utils.data import Dataset
from PIL import Image
from .augmentations import train_transforms,test_transforms
class ForensicsDataset(Dataset):
    def __init__(self,root,split='train'):
        self.items=[]; base=os.path.join(root,split)
        for lab in ['real','fake']:
            folder=os.path.join(base,lab)
            if not os.path.exists(folder): continue
            for f in os.listdir(folder): self.items.append((os.path.join(folder,f),0 if lab=='real' else 1))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): p,l=self.items[i]; img=Image.open(p).convert('RGB'); t=train_transforms if l in [0,1] else test_transforms; return t(img),l
