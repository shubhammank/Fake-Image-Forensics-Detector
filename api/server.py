from fastapi import FastAPI,UploadFile,File
import tempfile
from PIL import Image
from torchvision import transforms
from src.models.cnn_baseline import CNNBaseline
from src.models.uvfd_encoder import UVFDEncoder
from src.models.patch_transformer import PatchTransformer
from src.models.patch_scanner import PatchScanner
from src.models.recon_detector import ReconDetector
from src.inference.ensemble import AdvancedForensicEnsemble

app=FastAPI()
cnn=CNNBaseline(); uv=UVFDEncoder(); pt=PatchTransformer(); ps=PatchScanner(); rd=ReconDetector()
ens=AdvancedForensicEnsemble(cnn,uv,pt,ps,rd)
toT=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

@app.post('/detect')
async def detect(file:UploadFile=File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as t: t.write(await file.read()); path=t.name
    img=toT(Image.open(path).convert('RGB')).unsqueeze(0)
    return ens.predict(img)
