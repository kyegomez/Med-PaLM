import torch
from medpalm.model import MedPalm
from medpalm.model import ViTransformerWrapper, Transformer, Encoder, Decoder

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 4096))



model = MedPalm()
output = model(text, img)




# encoder = ViTransformerWrapper(
#     image_size = 256,
#     patch_size = 32,
#     attn_layers = Encoder(
#         dim = 512,
#         depth = 6,
#         heads = 8
#     )
# )

# decoder = Transformer(
#     num_tokens = 20000,
#     max_seq_len = 1024,
#     attn_layers = Decoder(
#         dim = 512,
#         depth = 6,
#         heads = 8,
#         cross_attend = True
#     ),
# )

# img = torch.randn(1, 3, 256, 256)
# caption = torch.randint(0, 20000, (1, 1024))

# encoded = encoder(img, return_embeddings = True)
# decoded = decoder(caption, context = encoded) # (1, 1024, 20000)
# print(decoded)
# print(decoded.shape)