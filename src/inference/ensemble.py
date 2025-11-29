import torch
class AdvancedForensicEnsemble:
    def __init__(self,cnn,uvfd,trans,scan,recon):
        self.cnn=cnn.eval(); self.uvfd=uvfd.eval(); self.trans=trans.eval(); self.scan=scan.eval(); self.recon=recon.eval()
    def predict(self,x):
        cnn=torch.softmax(self.cnn(x),1)[0,1]
        uv=torch.softmax(self.uvfd(x),1)[0,1]
        tr=torch.softmax(self.trans(x),1)[0,1]
        re=torch.sigmoid(1-self.recon(x)[0])
        p,coords=self.scan(x); ps=torch.softmax(p,1)[:,1].mean()
        score=float(0.25*cnn+0.25*uv+0.2*tr+0.15*ps+0.15*re)
        return {"label":"FAKE" if score>0.5 else "REAL","score":score}
