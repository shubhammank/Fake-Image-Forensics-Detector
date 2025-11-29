from PIL import Image
import torch
from torchvision import transforms
class Predictor:
    def __init__(self,model): self.m=model.eval(); self.t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    def predict(self,path): img=self.t(Image.open(path).convert('RGB')).unsqueeze(0); out=self.m(img); return 'FAKE' if out.argmax()==1 else 'REAL'
