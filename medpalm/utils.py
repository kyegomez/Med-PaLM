import math

import torch
import torch.distributed as dist  # Add this line
import torch.nn.functional as F
from einops import rearrange


def print_num_params(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if dist.is_available():
        if dist.get_rank() == 0:
            print(f"Number of parameters in model: {n_params}")
    else:
        print(f"Number of parameters in model: {n_params}")

def print_main(msg):
    if dist.is_available():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)


# helpers
def exists(val):
    return val is not None


#decorators
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

#tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def masked_mean(seq, mask=None, dim=1, keepdims=False):
    if not exists(mask):
        return seq.mean(dim=dim)
    
    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    
    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim=dim, keepdims=keepdims)
    denom = mask.sum(dim=dim, keepdims=keepdims)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom==0, 0.)
    return masked_mean

#sampling helpers
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs