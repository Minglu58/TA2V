import piq
import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm
import os
import torch
from skimage.metrics import structural_similarity as ssim_metric
#from torchmetrics.multimodal import CLIPScore
import torch.nn.functional as F
import clip
import torchvision.transforms.functional as T

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 100 * (x @ y.t()).squeeze()
    else:
        distance = 1 - (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance

    
def metrics_from_files(txt_files, video_files, resize, num_workers, print_256, idx, batch_size):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    #batch_size = 1
    total_size = len(video_files)

    if len(idx) == 0:
        clip_score = []
    else:
        clip_score = [[] for _ in idx]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    #metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    
    with torch.no_grad():
        for i in tqdm(range(total_size // batch_size)):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            videos_np = load_videos([video_files[i] for i in range(start, end)], resize, num_workers) #(1,16,112,112,3)
            videos = torch.tensor(videos_np).cuda() / 255
            
            # get text embedding
            text_emb = []
            for t in range(start, end):
                text = [line.rstrip() for line in open(txt_files[t])]  
                text = clip.tokenize(text).to(device)
                text_features = model.encode_text(text)
                text_emb.append(text_features)   
                       
            for bs in range(videos.shape[0]):
                dists = 0
                videos_i = videos[bs].permute(0,3,1,2)
                text_i = text_emb[bs]
                for idx in range(videos_i.shape[0]):
                    image = T.resize(videos_i[idx], [model.visual.input_resolution, model.visual.input_resolution]).unsqueeze(0)
                    image_embed = model.encode_image(image)
                    dist = d_clip_loss(image_embed, text_i, use_cosine=True) 
                    dists += dist / videos_i.shape[0]
                clip_score.append(dists.cpu().numpy()) 
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

def get_txt_files(folder):
    return sorted(glob(os.path.join(folder, '*.txt')))

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
    txt_tag = args.exp_tag if args.real_tag is None else args.txt_tag
    txt_folders = get_folders(txt_tag, args.num_folds)
    
    print('video_folders: ', video_folders)
    print('txt_folders: ', txt_folders)

    if len(args.idx) == 0:
        real_clip_score, fake1_clip_score, fake2_clip_score = [], [], []
    else:
        clip_score = [[] for _ in args.idx]
    for i, (txt_root, video_root) in tqdm(enumerate(zip(sorted(txt_folders), sorted(video_folders)))):
        
        print(f"[{i}] Loading txt")
        txt_files = get_txt_files(os.path.join(txt_root, args.txt_folder))
        print(f"Found {len(txt_files)} {args.txt_folder} files")
        
        print(f"[{i}] Loading real video")
        real_video_files = get_video_files(os.path.join(video_root, args.real_video_folder))
        print(f"Found {len(real_video_files)} {args.real_video_folder} video files")
        
        print(f"[{i}] Loading fake1 video")
        fake1_video_files = get_video_files(os.path.join(video_root, args.fake1_video_folder))
        print(f"Found {len(fake1_video_files)} {args.fake1_video_folder} video files")
        
        print(f"[{i}] Loading fake2 video")
        fake2_video_files = get_video_files(os.path.join(video_root, args.fake2_video_folder))
        print(f"Found {len(fake2_video_files)} {args.fake2_video_folder} video files")

        assert len(txt_files) == len(real_video_files) == len(fake1_video_files) == len(fake2_video_files)

        print(f"[{i}] Computing clip scores")
        real_clip_i = metrics_from_files(txt_files, real_video_files, args.resize, args.num_workers, args.print_256, args.idx, args.batch_size)
        fake1_clip_i = metrics_from_files(txt_files, fake1_video_files, args.resize, args.num_workers, args.print_256, args.idx, args.batch_size)
        fake2_clip_i = metrics_from_files(txt_files, fake2_video_files, args.resize, args.num_workers, args.print_256, args.idx, args.batch_size)
        
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
    parser.add_argument('--real_tag', type=str, default=None)
    parser.add_argument('--txt_folder', type=str, default="txt")
    parser.add_argument('--real_video_folder', type=str, default="real")
    parser.add_argument('--fake1_video_folder', type=str, default="fake_stage1")
    parser.add_argument('--fake2_video_folder', type=str, default="fake_stage2")
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--idx', type=int, nargs="+", default=[])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--print_256', action='store_true')
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args)
