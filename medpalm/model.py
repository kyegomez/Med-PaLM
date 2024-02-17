import torch
from transformers import AutoTokenizer, CLIPProcessor

from medpalm.transformer import (
    AutoregressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)


class MedPalmTokenizer:
    def __init__(self):
        try:
            self.processor = CLIPProcessor.from_pretrained(
                "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192,
            )

            self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(
                ["<image>", "</image>"]
            )
        except Exception as e:
            print(f"Error init tokenizer: {e}")

    def tokenize_texts(self, texts):
        try:
            texts = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).input_ids
            image_tokens = torch.tensor(
                [[self.im_idx, self.im_end_idx]] * texts.shape[0]
            )
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            print(f"Error tokenizing texts: {e}")

    def tokenize_images(self, images):
        try:
            tokenized_images = self.processor(
                images=images, return_tensors="pt"
            ).pixel_values
            print(f"Tokenized image: {tokenized_images.shape}")
            return tokenized_images

        except Exception as e:
            print(f"Error tokenizing texts: {e}")

    def tokenize(self, sample):
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }

        except Exception as e:
            print(f"Error during tokenization {e}")


class MedPalm(torch.nn.Module):
    """
    MedPalm model for medical image and text processing.

    Args:
        image_size (int): Size of the input image (default: 256).
        patch_size (int): Size of each image patch (default: 32).
        encoder_dim (int): Dimensionality of the encoder (default: 512).
        encoder_depth (int): Number of encoder layers (default: 6).
        encoder_heads (int): Number of attention heads in the encoder (default: 8).
        num_tokens (int): Number of tokens in the decoder (default: 20000).
        max_seq_len (int): Maximum sequence length in the decoder (default: 1024).
        decoder_dim (int): Dimensionality of the decoder (default: 512).
        decoder_depth (int): Number of decoder layers (default: 6).
        decoder_heads (int): Number of attention heads in the decoder (default: 8).
        alibi_num_heads (int): Number of attention heads in the alibi mechanism (default: 4).
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings (default: False).
        cross_attend (bool): Whether to enable cross-attention in the decoder (default: True).
        alibi_pos_bias (bool): Whether to use positional bias in the alibi mechanism (default: True).
        rotary_xpos (bool): Whether to use rotary positional embeddings (default: True).
        attn_flash (bool): Whether to use attention flash in the decoder (default: True).
        qk_norm (bool): Whether to normalize the query-key vectors in attention (default: True).
    """

    def __init__(
        self,
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        use_abs_pos_emb=False,
        cross_attend=True,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(MedPalm, self).__init__()

        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )

        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, img, text):
        """
        Forward pass of the MedPalm model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
