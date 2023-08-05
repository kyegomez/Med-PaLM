
import bitsandbytes
import torch
import torch.nn as nn
from flamingo_pytorch import PerceiverResampler
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from med_palm.palm import PaLM


class MedPalmTokenizer:
    def __init__(self):
        try:

            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token ="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )

            self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])
        except Exception as e:
            print(f"Error init tokenizer: {e}")


    def tokenize_texts(self, texts):
        try:

            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            print(f"Error tokenizing texts: {e}")

        

    def tokenize_images(self, images):
        try:
                
            tokenized_images = self.processor(images=images, return_tensors="pt").pixel_values
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
        
class MedPalm(nn.Module):
    def __init__(self):
        super(MedPalm, self).__init__()
        try:

            self.ViT_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model

            self.embed = bitsandbytes.nn.modules.Embedding(
                32002,
                2048,
                padding_idx=1
            )

            self.output_projection = torch.nn.Linear(
                2048, 32002, bias=False
            )
            torch.nn.init.normal_(
                self.output_projection.weight, mean=0, std=2048**-0.5
            )

            self.decoder = PaLM(
                num_tokens=50304,
                dim=2048,
                depth=16,
                dim_head=128,
                heads=8,
                flash_attn=True,
                qk_rmsnorm=False,
            )

            self.perceive = PerceiverResampler(
                dim= 1024,
                depth = 2,
                dim_head = 8,
                num_latents = 64,
                num_media_embeds = 257
            )

            self.image_proj = torch.nn.Linear(1024, 2048, bias=False)
            torch.nn.init.normal_(
                self.image_proj.weight, mean=0, std=2048**-0.5
            )

        except Exception as e:
            print(f"Error initlizing palme components: {e}")

    def forward(self, text_tokens, images):
        try:
                
            # images = images.view(images.size(0), -1)  # Flatten the images
            # images = images.view(images.size(0), 3, 1024, 1024)  # Reshape the images to the expected size
            # images = self.perceive(images).squeeze(1)
            # images = self.image_proj(images)
            # images_flattened = images.view(images.size(0), -1)
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)

            model_input = self.decoder(text_tokens)

            # images_flattened = images_flattened.view(1, 2, -1)

            model_input = torch.cat([model_input[:, 0:2], images, model_input[:, 2:]], dim=-1)

            model_input = self.decoder(model_input, tokens_mask=None)

            output = self.decoder(model_input, passed_x=model_input)[0]

            return output
        
        except Exception as e:
            print(f"Error duing forward pass: {e}")
            return None