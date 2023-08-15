# PALM-E
Implementation of "PaLM-E: An Embodied Multimodal Language Model"


## Summary:

PALME is an embodied multimodal language model that injects continuous, embodied observations such as images, state estimates, or other sensor modalities into the language embedding space of a pre-trained language model. It uses PaLM as the pre-trained language model and makes it embodied. 

The inputs to PALME consist of text and continuous observations, which are interleaved to form multimodal sentences. The output is text generated autoregressively by the model, which could be an answer to a question or a sequence of decisions for a robot.

### Key Components:

Decoder-only LLMs: Large language models that generate text based on a given prefix or prompt.

Token embedding space: Maps tokens from a fixed vocabulary into a word token embedding space.

Multi-modal sentences: Inject continuous observations into the LLM by mapping them directly into the language embedding space.

Embodying the output: Connect the output of the model to an embodiment, either as text or as a high-level policy for robot control.

Input & Scene Representations: Different encoders for various sensor modalities, such as state estimation vectors, Vision Transformers (ViTs), and Object Scene Representation Transformer (OSRT).

Training Recipes: Train PALME on a dataset with continuous observations and text, using cross-entropy loss for non-prefix tokens. Experiment with model freezing and co-training across tasks.


### Mathematical Architecture
Language Model: p(w1:L) = ∏L l=1 pLM(wl | w1:l−1)
Prefix-decoder-only LLM: p(wn+1:L | w1:n) = ∏L l=n+1 pLM(wl | w1:l−1)
Token embedding space: γ : W → X
Multi-modal sentences: xi = { γ(wi) if i is a text token, or φj(Oj)i if i corresponds to observation Oj }
Encoder: φ : O → X^q
Projector: ψ : R^k˜ → R^m×k

## Algorithmic Pseudocode
```python 
initialize LLM, encoder, projector

for each example in dataset:
    extract continuous observations, text, and index
    for each continuous observation:
        encode observation using encoder
    for each token in text:
        if token is a text token:
            embed token using LLM's token embedding space
        else:
            replace token with encoded observation
    compute loss using cross-entropy for non-prefix tokens
    update model parameters
```

Algorithmic PyTorch Implementation

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class PALME(nn.Module):
    def __init__(self, LLM, encoder, projector):
        super(PALME, self).__init__()
        self.LLM = LLM
        self.encoder = encoder
        self.projector = projector

    def forward(self, continuous_observations, text, index):
        embeddings = []
        for token in text:
            if token_is_text(token):
                embeddings.append(self.LLM.embed(token))
            else:
                observation = continuous_observations[token.observation_index]
                encoded_observation = self.encoder(observation)
                embeddings.append(self.projector(encoded_observation))
        embeddings = torch.stack(embeddings)
        output = self.LLM(embeddings)
        return output

def train_PALME(model, dataset, epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for continuous_observations, text, index in dataset:
            optimizer.zero_grad()
            output = model(continuous_observations, text, index)
            loss = criterion(output, text[index:])
            loss.backward()
            optimizer.step()
```
## Math
Pre-trained Language Model (PaLM)
Encoder: φ : O → X^q (for continuous observations)
Projector: ψ : R^k˜ → R^m×k (for mapping encoder output to language embedding space)

### Algorithmic Overview
Initialize PaLM, encoder, and projector.
For each example in the dataset:
Extract text and continuous observations.
For each continuous observation:
Encode observation using the encoder.
Interleave encoded observations with text tokens to form multi-modal sentences.
Compute loss using cross-entropy for non-prefix tokens.
Update model parameters.

# Detailed Steps
Initialize the pre-trained Language Model (PaLM) as the base model.
Choose an encoder for continuous observations:
State estimation vectors
Vision Transformer (ViT)
Object Scene Representation Transformer (OSRT)
Initialize a projector to map the encoder output to the language embedding space.
For each example in the dataset: a. Extract text and continuous observations. b. Encode continuous observations using the chosen encoder. c. Interleave encoded observations with text tokens to form multi-modal sentences. d. Compute loss using cross-entropy for non-prefix tokens. e. Update model parameters.
