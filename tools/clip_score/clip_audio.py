import piq
import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
import clip
import wav2clip
import librosa
from tools import AudioCLIP


def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 100 * (x @ y.t()).squeeze()
    else:
        distance = 1 - (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def audioclip_loss(x, y, model):
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
    y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
    
    #aclp = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
    scale_audio_image = torch.clamp(model.logit_scale_ai.exp(), min=1.0, max=100.0)
    
    distance = scale_audio_image * x @ y.T
    
    return distance


def calculate_wav2clip_score(audio_files, video_files, resize, num_workers, print_256, sequence_length, idx, batch_size):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    #batch_size = 1
    total_size = len(video_files)

    if len(idx) == 0:
        clip_score = []
    else:
        clip_score = [[] for _ in idx]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    wav2clip_model = wav2clip.get_model()
    for p in wav2clip_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for i in tqdm(range(total_size // batch_size)):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            videos_np = load_videos([video_files[i] for i in range(start, end)], resize, num_workers) #(1,16,112,112,3)
            videos = torch.tensor(videos_np).cuda() / 255
          
            audio_emb = None
            for t in range(start, end):
                audio, sr = librosa.load(audio_files[t], sr=48000)
                audio = audio.reshape(sequence_length, -1)
                audio_features = torch.from_numpy(wav2clip.embed_audio(audio, wav2clip_model)).cuda().unsqueeze(0)
                if audio_emb is None:
                    audio_emb = audio_features
                else:
                    audio_emb = torch.cat((audio_emb, audio_features), axis=0)
         
            for bs in range(videos.shape[0]):
                dists = 0
                videos_i = videos[bs].permute(0,3,1,2)
                audio_i = audio_emb[bs]
                for idx in range(videos_i.shape[0]):
                    image = T.resize(videos_i[idx], [model.visual.input_resolution, model.visual.input_resolution]).unsqueeze(0)
                    image_embed = model.encode_image(image).float()
                    dist = d_clip_loss(image_embed, audio_i[idx,:], use_cosine=True) 
                    dists += dist / videos_i.shape[0]
                clip_score.append(dists.cpu().numpy()) 
    print(clip_score)
    print(np.std(clip_score))
    return np.mean(clip_score)

def calculate_audioclip_score(audio_files, video_files, resize, num_workers, print_256, sequence_length, idx, batch_size):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    #batch_size = 1
    total_size = len(video_files)

    if len(idx) == 0:
        clip_score = []
    else:
        clip_score = [[] for _ in idx]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    audioclip = AudioCLIP(pretrained=f'TAV/saved_ckpts/AudioCLIP-Full-Training.pt')
    with torch.no_grad():
        for i in tqdm(range(total_size // batch_size)):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            videos_np = load_videos([video_files[i] for i in range(start, end)], resize, num_workers) #(1,16,112,112,3)
            videos = torch.tensor(videos_np).cuda() / 255
          
            audio_emb = None
            for t in range(start, end):
                audio, sr = librosa.load(audio_files[t], sr=48000)
                audio = audio.reshape(sequence_length, -1)
                ((audio_features, _, _), _), _ = audioclip(audio=torch.tensor(audio))
                audio_features = audio_features.cuda()
                if audio_emb is None:
                    audio_emb = audio_features
                else:
                    audio_emb = torch.cat((audio_emb, audio_features), axis=0)
                  
            for bs in range(videos.shape[0]):
                videos_i = videos[bs].permute(0,3,1,2)
                image = T.resize(videos_i, [model.visual.input_resolution, model.visual.input_resolution])
                ((_, image_features, _), _), _ = audioclip(image=image.cpu())
                dist = audioclip_loss(audio_emb, image_features.cuda(), audioclip)
                clip_score.append(torch.diag(dist).cpu().numpy().mean()) 
    print(clip_score)
    print(np.std(clip_score))
    return np.mean(clip_score)
            

def load_video(file, resize):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    while success:
        if resize is not None:
            h, w = resize
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(image)
        success, image = vidcap.read()
    return np.stack(frames)

def get_video_files(folder):
    return sorted(glob(os.path.join(folder, '*.mp4')))

def get_audio_files(folder):
    return sorted(glob(os.path.join(folder, '*.wav')))

def load_videos(video_files, resize, num_workers):
    videos = Parallel(n_jobs=num_workers)(delayed(load_video)(file, resize) for file in video_files)
    return np.stack(videos)

def get_folder(exp_tag, fold_i=None):
    if fold_i is not None:
        exp_tag = f"{fold_i}_" + exp_tag
    all_folders = glob(f"./results/{exp_tag}*")

    assert len(all_folders) == 1, f"Too many possibilities for this tag {exp_tag}:\n{all_folders}"
    return all_folders[0]

def get_folders(exp_tag, num_folds):
    if num_folds is not None:
        folders= []
        for i in range(num_folds):
            folders.append(get_folder(exp_tag, i))
        return folders
    else:
        return [get_folder(exp_tag)]

def upscale(videos, min_size=96):
    h, w = videos.shape[-2:]
    if h >= min_size and w >= min_size:
        return videos
    else:
        if h < w:
            size = [min_size, int(min_size * w / h)]
        else:
            size = [int(min_size * h / w), min_size]
        return torch.nn.functional.interpolate(videos, size=size, mode='bilinear')

def print_scores(scores, name):
    print(f"Individual {name} scores")
    print(scores)
    print(f"Mean/std of {name} across {len(scores)} runs")
    print(np.mean(scores), np.around(np.std(scores), decimals=3))

def main(args):
    video_folders = get_folders(args.exp_tag, args.num_folds)
    audio_tag = args.exp_tag if args.audio_tag is None else args.audio_tag
    audio_folders = get_folders(audio_tag, args.num_folds)
    
    print('video_folders: ', video_folders)
    print('audio_folders: ', audio_folders)

    if len(args.idx) == 0:
        real_clip_score, fake1_clip_score, fake2_clip_score = [], [], []
    else:
        clip_score = [[] for _ in args.idx]
    for i, (audio_root, video_root) in tqdm(enumerate(zip(sorted(audio_folders), sorted(video_folders)))):
        
        print(f"[{i}] Loading audio")
        audio_files = get_audio_files(os.path.join(audio_root, args.audio_folder))
        print(f"Found {len(audio_files)} {args.audio_folder} files")

        print(f"[{i}] Loading real video")
        real_video_files = get_video_files(os.path.join(video_root, args.real_video_folder))
        print(f"Found {len(real_video_files)} {args.real_video_folder} video files")
        
        print(f"[{i}] Loading fake1 video")
        fake1_video_files = get_video_files(os.path.join(video_root, args.fake1_video_folder))
        print(f"Found {len(fake1_video_files)} {args.fake1_video_folder} video files")
        
        print(f"[{i}] Loading fake2 video")
        fake2_video_files = get_video_files(os.path.join(video_root, args.fake2_video_folder))
        print(f"Found {len(fake2_video_files)} {args.fake2_video_folder} video files")
        

        assert len(audio_files) == len(real_video_files) == len(fake1_video_files) == len(fake2_video_files)

        print(f"[{i}] Computing clip scores")
        if args.audio_emb_model == 'wav2clip':
            real_clip_i = calculate_wav2clip_score(audio_files, real_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
            fake1_clip_i = calculate_wav2clip_score(audio_files, fake1_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
            fake2_clip_i = calculate_wav2clip_score(audio_files, fake2_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
        
        elif args.audio_emb_model == 'audioclip':
            real_clip_i = calculate_audioclip_score(audio_files, real_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
            fake1_clip_i = calculate_audioclip_score(audio_files, fake1_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
            fake2_clip_i = calculate_audioclip_score(audio_files, fake2_video_files, args.resize, args.num_workers, args.print_256, args.sequence_length, args.idx, args.batch_size)
        
        if len(args.idx) == 0:
            real_clip_score.append(real_clip_i)
            fake1_clip_score.append(fake1_clip_i)
            fake2_clip_score.append(fake2_clip_i)
        else:
            for k in range(len(args.idx)):
                clip_score[k].append(clip_i[k])

    if len(args.idx) == 0:
        print_scores(real_clip_score, "Audio-RealVideo CLIP")
        print_scores(fake1_clip_score, "Audio-Fake1Video CLIP")
        print_scores(fake2_clip_score, "Audio-Fake2Video CLIP")
    else:
        for k in range(len(args.idx)):
            print_scores(clip_score[k], f"CLIP-{k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_tag', type=str, default=None)
    parser.add_argument('--audio_tag', type=str, default=None)
    parser.add_argument('--audio_folder', type=str, default="audio")
    parser.add_argument('--real_video_folder', type=str, default="real")
    parser.add_argument('--fake1_video_folder', type=str, default="fake_stage1")
    parser.add_argument('--fake2_video_folder', type=str, default="fake_stage2")
    parser.add_argument('--audio_emb_model', type=str, default='wav2clip', choices=['wav2clip', 'audioclip'])
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--idx', type=int, nargs="+", default=[])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--print_256', action='store_true')
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args)
