import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.logger import log

def train_model(model,dataset,lr=1e-4,epochs=1):
    loader=DataLoader(dataset,batch_size=4,shuffle=True); opt=torch.optim.Adam(model.parameters(),lr); loss_fn=nn.CrossEntropyLoss()
    for e in range(epochs): tot=0
    return model
