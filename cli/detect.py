import argparse
from src.models.cnn_baseline import CNNBaseline
from src.inference.predict import Predictor
p=argparse.ArgumentParser(); p.add_argument('--image',required=True); a=p.parse_args(); m=CNNBaseline(pretrained=False); pr=Predictor(m); print(pr.predict(a.image))