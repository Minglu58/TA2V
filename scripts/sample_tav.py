import os
import tqdm
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import librosa
import soundfile

from tav import VideoData, load_transformer, load_text_transformer
from tav.utils import save_video_grid
from tav.utils import shift_dim
from tav.modules.gpt import sample_with_past

from diffusion import logger
from optimization.image_editor import ImageEditor
from optimization.video_editor import VideoEditor
from optimization.arguments import get_arguments

from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_text_ckpt', type=str, default='')
parser.add_argument('--vqgan_ckpt', type=str, default='')
parser.add_argument('--stft_vqgan_ckpt', type=str, default='')

parser.add_argument('--save', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/post_URMP/')
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.80)
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--dataset', type=str, default='URMP', choices=['URMP', 'landscape'])
parser.add_argument('--compute_fvd', action='store_true')
parser.add_argument('--text_stft_cond', action='store_true')
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--text_emb_model', type=str, default=None, choices=['bert', 'clip'])
parser.add_argument('--audio_emb_model', type=str, default='wav2clip', choices=['STFT', 'wav2clip', 'audioclip'])
parser.add_argument('--resolution', type=int, default=128)
parser = get_arguments(parser)
args = parser.parse_args()

gpt_text = load_text_transformer(args.gpt_text_ckpt, args.vqgan_ckpt, args.stft_vqgan_ckpt).cuda().eval()

gpt_text.args.batch_size = 1
gpt_text.args.data_path = args.data_path
gpt_text.args.sequence_length = args.sequence_length
gpt_text.args.audio_emb_model = args.audio_emb_model
gpt_text.args.resolution = args.resolution
data = VideoData(gpt_text.args)
loader = data.test_dataloader()

@torch.no_grad()
def sample(model, batch_size, cond, steps=256, temperature=None, top_k=None, callback=None,
                            verbose_time=False, top_p=None, latent_shape=(4, 12, 12), n_cond=0):
    log = dict()
    t1 = time.time()
    index_sample = sample_with_past(cond, model.transformer_text, steps=steps,
                                    sample_logits=True, top_k=top_k, callback=callback,
                                    temperature=temperature, top_p=top_p)
        
    if verbose_time:
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        
    index = index_sample.reshape([batch_size, *latent_shape])
    index = torch.clamp(index-n_cond, min=0, max=model.first_stage_model.n_codes-1)
    # x_sample = torch.cat([model.first_stage_model.decode(index[:batch_size//2]-n_cond), model.first_stage_model.decode(index[batch_size//2:]-n_cond)])
    
    x_sample = model.first_stage_model.decode(index)
    log["samples"] = torch.clamp(x_sample, -0.5, 0.5)
    return log

#save_dir = '%s/%s_groundtruth_3modalities/%s/topp%.2f_topk%d'%(args.save, args.run, args.dataset, args.top_p, args.top_k)
#print('generating and saving video to %s...'%save_dir)
#os.makedirs(save_dir, exist_ok=True)


steps = np.prod(gpt_text.first_stage_model.latent_shape)
print('Total steps: ', str(steps)) #576


with torch.no_grad():
    for sample_id in tqdm.tqdm(range(min(args.n_sample, len(loader.dataset)))):
        logs = dict()   
        batch = loader.dataset.__getitem__(sample_id)
        #load data
        
        x, c1 = gpt_text.get_xc(batch)
        os.makedirs('./results/%d_tav_%s/real/'%(args.run, args.dataset), exist_ok=True)
        save_video_grid(torch.clamp(x.unsqueeze(0), -0.5, 0.5) + 0.5, os.path.join('./results/%d_tav_%s/real/'%(args.run, args.dataset), 'groundtruth_%s_%d.mp4'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)), 1, fps=30)
        
        _, c1_indices = gpt_text.encode_to_c1(c1)
        c1_indices = c1_indices.cuda()
        
        #stage1 sampling
        logger.log("creating samples using text...")
        logs = sample(gpt_text, batch_size=1, cond=c1_indices, steps=steps, n_cond=gpt_text.cond1_stage_vocab_size, 
                      temperature=1., top_k=args.top_k, top_p=args.top_p, verbose_time=True, latent_shape=gpt_text.first_stage_model.latent_shape)
        
        #preprocess videos before editing
        video = logs["samples"].squeeze()
        video = F.interpolate(video, size=(args.model_output_size, args.model_output_size), mode='bilinear',align_corners=False)
        cond = batch['audio']
        text = batch['raw_text']
        
        #stage2 editing
        logger.log("creating video editor...")       
        video_editor = VideoEditor(args)
        pred_video = video_editor.edit_video_by_prompt(video, cond, text)
        
        # save stage1 video
        os.makedirs('./results/%d_tav_%s/fake_stage1/'%(args.run, args.dataset), exist_ok=True)
        save_video_grid(logs["samples"]+0.5, os.path.join('./results/%d_tav_%s/fake_stage1/'%(args.run, args.dataset), 'generation_stage1_%s_%d.mp4'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)), 1,fps=30)
       
        # save stage2 video                                           
        os.makedirs('./results/%d_tav_%s/fake_stage2/'%(args.run, args.dataset), exist_ok=True)  
        save_video_grid(pred_video, os.path.join('./results/%d_tav_%s/fake_stage2/'%(args.run, args.dataset), 'generation_stage2_%s_%d.mp4'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)), 1, fps=30)
        
        os.makedirs('./results/%d_tav_%s/txt/'%(args.run, args.dataset), exist_ok=True)
        os.makedirs('./results/%d_tav_%s/audio/'%(args.run, args.dataset), exist_ok=True)
        batch['audio'] = batch['audio'].reshape(-1).numpy()
        soundfile.write(os.path.join('./results/%d_tav_%s/audio/'%(args.run, args.dataset), 'groundtruth_%s_%d.wav'%(os.path.basename(batch['path'])[:-len('.mp4')],sample_id)), batch['audio'], 48000)
        
        #copyfile(batch['path'], os.path.join(save_dir, 'groundtruth_%s_%d.mp4'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)))
        copyfile(batch['path'].replace("/mp4/", "/txt/").replace(".mp4", ".txt"), os.path.join('./results/%d_tav_%s/txt/'%(args.run, args.dataset), 'groundtruth_%s_%d.txt'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)))