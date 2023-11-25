# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import requests
from tqdm import tqdm
import os
import torch

from .vqgan import VQGAN
from .transformer import Net2NetTransformer
from .text_transformer import TextTransformer
from .stft_transformer import AudioTransformer

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id, fname, root=os.path.expanduser('./ckpts')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


def load_vqgan(vqgan_ckpt, device=torch.device('cpu')):
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).to(device)
    vqgan.eval()

    return vqgan

def load_transformer(gpt_ckpt, vqgan_ckpt, stft_vqgan_ckpt='', device=torch.device('cpu')):
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    checkpoint = pl_load(gpt_ckpt)
    checkpoint['hyper_parameters']['args'].vqvae = vqgan_ckpt
    if stft_vqgan_ckpt:
        checkpoint['hyper_parameters']['args'].stft_vqvae = stft_vqgan_ckpt
    gpt = Net2NetTransformer._load_model_state(checkpoint)
    gpt.eval()

    return gpt

def load_text_transformer(gpt_ckpt, vqgan_ckpt, stft_vqgan_ckpt='', device=torch.device('cpu')):
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    checkpoint = pl_load(gpt_ckpt)
    checkpoint['hyper_parameters']['args'].vqvae = vqgan_ckpt
    if stft_vqgan_ckpt:
        checkpoint['hyper_parameters']['args'].stft_vqvae = stft_vqgan_ckpt
    gpt = TextTransformer._load_model_state(checkpoint)
    gpt.eval()

    return gpt

def load_audio_transformer(gpt_ckpt, vqgan_ckpt, stft_vqgan_ckpt='', device=torch.device('cpu')):
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    checkpoint = pl_load(gpt_ckpt)
    checkpoint['hyper_parameters']['args'].vqvae = vqgan_ckpt
    if stft_vqgan_ckpt:
        checkpoint['hyper_parameters']['args'].stft_vqvae = stft_vqgan_ckpt
    gpt = AudioTransformer._load_model_state(checkpoint)
    gpt.eval()

    return gpt
