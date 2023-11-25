from torch.nn import functional as F
import torch
from tools import AudioCLIP

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def audioclip_loss(x, y, model, use_scale=False, use_cosine=False):
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
    y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
    
    #aclp = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
    scale_audio_image = torch.clamp(model.logit_scale_ai.exp(), min=1.0, max=100.0)
    
    if use_scale:
        distance = scale_audio_image * (1 -  x @ y.T)
    else:
        if use_cosine:
            distance = 1 -  x @ y.T
        else:
            distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    
    return distance
