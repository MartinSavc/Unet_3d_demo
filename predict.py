from pyexpat import model
import torch
from model import Unet3D


# primer za predikcijo z shranjenim naučenim modelom
if __name__ == '__main__':
    model = Unet3D()
    model_state_dict = torch.load('model_last.pth')
    model.load_state_dict(model_state_dict)
    
    model.eval()
    
    with torch.no_grad():
        # tukaj bi nalozili prave podatke
        # le ti lahko imajo drugačne prostorske dimenzije kot učni, kadar uporabljamo konvolucijske mreže
        # je pa dobro, da so deljivi z 2^n, kjer je n število ločljivnostnih nivojev v mreži
        x = torch.randn(1, 1, 64, 64, 64)

        y = model(x)