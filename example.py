import torch
from medpalm.model import MedPalm
from medpalm.model import ViTransformerWrapper, Transformer, Encoder, Decoder

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 4096))



model = MedPalm()
output = model(text, img)


