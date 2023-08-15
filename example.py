import torch
from medpalm.model import MedPalm

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = MedPalm()
output = model(img, caption_tokens)