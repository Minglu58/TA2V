# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import h5py
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl

from .coinrun.coinrun_data import CoinRunDataset
from .coinrun.tokenizer import tokenizer
# from transformers import BertTokenizer


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64, sample_every_n_frames=1):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)

        # self._clips = clips.subset(np.arange(24))
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        while True:
            try:
                video, _, _, idx = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(**preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames), label=label)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None, in_channels=3, sample_every_n_frames=1):
    # video: THWC, {0, ..., 255}
    if in_channels == 3:
        video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW
    else:
        # make the semantic map one hot
        if video.shape[-1] == 3:
            video = video[:, :, :, 0]
        video = F.one_hot(video.long(), num_classes=in_channels).permute(0, 3, 1, 2).float()
        # flatseg = video.reshape(-1)
        # onehot = torch.zeros((flatseg.shape[0], in_channels))
        # onehot[torch.arange(flatseg.shape[0]), flatseg] = 1
        # onehot = onehot.reshape(video.shape + (in_channels,))
        # video = onehot.permute(0, 3, 1, 2).float()
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # skip frames
    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    video -= 0.5
    if in_channels == 3:
        return {'video': video}
    else:
        return {'video_smap': video}


class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """

    def __init__(self, data_file, sequence_length, train=True, resolution=64, image_channels=3, sample_every_n_frames=1):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.image_channels = image_channels
        self.sample_every_n_frames = sample_every_n_frames

        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx'][:-1]
        self.size = len(self._idx)

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx'][:-1]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0
        # print(end, start, self._idx[idx + 1])
        start = start + torch.randint(low=0, high=end - start - self.sequence_length, size=(1,)).item()
        # start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        video = torch.tensor(self._images[start:start + self.sequence_length])
        return dict(**preprocess(video, self.resolution, sample_every_n_frames=self.sample_every_n_frames))


class VideoData(pl.LightningDataModule):

    def __init__(self, args, shuffle=True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        # check if it's coinrun dataset (path contains coinrun and it's a directory)
        if osp.isdir(self.args.data_path) and 'coinrun' in self.args.data_path.lower():
            if hasattr(self.args, 'coinrun_v2_dataloader') and self.args.coinrun_v2_dataloader:
                Dataset = CoinRunDatasetV2
            else:
                Dataset = CoinRunDataset
            if hasattr(self.args, 'smap_cond') and self.args.smap_cond:
                dataset = Dataset(data_folder=self.args.data_path, args=self.args, train=train, get_seg_map=True)
            elif hasattr(self.args, 'text_cond') and self.args.text_cond:
                if self.args.smap_only:
                    dataset = Dataset(data_folder=self.args.data_path, args=self.args, train=train,
                                      get_game_frame=False, get_seg_map=True, get_text_desc=True)
                else:
                    dataset = Dataset(data_folder=self.args.data_path, args=self.args, train=train, get_text_desc=True)
            elif self.args.smap_only:
                dataset = Dataset(data_folder=self.args.data_path, args=self.args, train=train,
                                  get_game_frame=False, get_seg_map=True)
            else:
                dataset = Dataset(data_folder=self.args.data_path, args=self.args, train=train)
        else:
            if hasattr(self.args, 'vtokens') and self.args.vtokens:
                Dataset = HDF5Dataset_vtokens
                dataset = Dataset(self.args.data_path, self.args.sequence_length,
                                  train=train, resolution=self.args.resolution, spatial_length=self.args.spatial_length,
                                  sample_every_n_frames=self.args.sample_every_n_frames)
            elif hasattr(self.args, 'image_folder') and self.args.image_folder:
                Dataset = FrameDataset
                dataset = Dataset(self.args.data_path, self.args.sequence_length,
                                  resolution=self.args.resolution, sample_every_n_frames=self.args.sample_every_n_frames)
            elif hasattr(self.args, 'stft_data') and self.args.stft_data:
                Dataset = StftDataset
                dataset = Dataset(self.args.data_path, self.args.sequence_length, train=train,
                                  sample_every_n_frames=self.args.sample_every_n_frames)
            elif hasattr(self.args, 'smap_cond') and self.args.smap_cond:
                Dataset = HDF5Dataset_smap
                dataset = Dataset(self.args.data_path, self.args.data_path2, self.args.sequence_length,
                                  train=train, resolution=self.args.resolution,
                                  image_channels1=self.args.image_channels1,
                                  image_channels2=self.args.image_channels2)
            elif hasattr(self.args, 'text_cond') and self.args.text_cond:
                Dataset = HDF5Dataset_text
                dataset = Dataset(self.args.data_path, self.args.sequence_length, self.args.text_emb_model,
                                  train=train, resolution=self.args.resolution, image_channels=self.args.image_channels,
                                  text_len=self.args.text_seq_len, truncate_captions=self.args.truncate_captions)
            elif hasattr(self.args, 'sample_every_n_frames') and self.args.sample_every_n_frames>1:
                Dataset = VideoDataset if osp.isdir(self.args.data_path) else HDF5Dataset
                dataset = Dataset(self.args.data_path, self.args.sequence_length,
                                  train=train, resolution=self.args.resolution, sample_every_n_frames=self.args.sample_every_n_frames)
            else:
                Dataset = VideoDataset if osp.isdir(self.args.data_path) else HDF5Dataset
                dataset = Dataset(self.args.data_path, self.args.sequence_length,
                                  train=train, resolution=self.args.resolution)
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            if hasattr(self.args, 'balanced_sampler') and self.args.balanced_sampler and train:
                sampler = BalancedRandomSampler(dataset.classes_for_sampling)
            else:
                sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle is True
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, default='/datasets01/Kinetics400_Frames/videos')
        parser.add_argument('--sequence_length', type=int, default=16)
        parser.add_argument('--resolution', type=int, default=64)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--image_channels', type=int, default=3)
        parser.add_argument('--smap_cond', type=int, default=0)
        parser.add_argument('--smap_only', action='store_true')
        parser.add_argument('--text_cond', action='store_true')
        parser.add_argument('--vtokens', action='store_true')
        parser.add_argument('--vtokens_pos', action='store_true')
        parser.add_argument('--spatial_length', type=int, default=15)
        parser.add_argument('--sample_every_n_frames', type=int, default=1)
        parser.add_argument('--image_folder', action='store_true')
        parser.add_argument('--stft_data', action='store_true')

        return parser

        
class HDF5Dataset_smap(data.Dataset):
    def __init__(self, data_file, data_file_cond, sequence_length, train=True, resolution=64, image_channels1=3,
                 image_channels2=66):
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.image_channels1 = image_channels1
        self.image_channels2 = image_channels2

        # read in data
        self.data = h5py.File(data_file, 'r')
        prefix = 'train' if train else 'test'
        self._images = self.data[f'{prefix}_data']
        self._idx = self.data[f'{prefix}_idx']

        # read in data
        self.data2 = h5py.File(data_file_cond, 'r')
        self._images2 = self.data2[f'{prefix}_data']

        # compute splits for all possible sequences
        self._splits = self._compute_seq_splits()

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def _compute_seq_splits(self):
        splits = []
        n_videos = len(self._idx)
        for i in range(n_videos - 1):
            start = self._idx[i]
            # end = self._idx[i + 1] if i < n_videos - 1 else n_videos
            end = self._idx[i + 1]
            splits.extend([(start + i, start + i + self.sequence_length)
                           for i in range(end - start - self.sequence_length + 1)])
        return splits

    def __len__(self):
        return len(self._splits)

    def __getitem__(self, idx):
        start_idx, end_idx = self._splits[idx]
        video = torch.tensor(self._images[start_idx:end_idx])
        video2 = torch.tensor(self._images2[start_idx:end_idx])
        return dict(**preprocess(video, self.resolution, in_channels=self.image_channels1),
                    **preprocess(video2, self.resolution, in_channels=self.image_channels2))


class HDF5Dataset_text(data.Dataset):
    def __init__(self, data_file, sequence_length, text_emb_model, train=True, resolution=64, image_channels=3,
                 text_len=256, truncate_captions=False):
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.image_channels = image_channels
        self.prefix = 'train' if train else 'test'
        self.text_len = text_len
        self.truncate_captions = truncate_captions

        # read in video data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx'][:-1]
        self.size = len(self._idx)
        self.text_emb_model = text_emb_model

        # read in text data
        self.text_file = os.path.join(os.path.dirname(data_file), '%s_text_description.txt' % self.prefix)
        self._text_annos = [line.rstrip() for line in open(self.text_file)]

        if text_emb_model == 'bert':
            print('using bert pretrain model...')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        state['_text_annos'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx'][:-1]
        self._text_annos = [line.rstrip() for line in open(self.text_file)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0
        # print(end, start, self._idx[idx + 1])

        start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        video = torch.tensor(self._images[start:start + self.sequence_length])
        if self.text_emb_model == 'bert':
            tokenized_text = self.tokenizer.encode(np.random.choice(self._text_annos[idx].split('\t')),
                                                   padding='max_length',
                                                   max_length=self.text_len,
                                                   truncation=self.truncate_captions,
                                                   return_tensors='pt').squeeze()
        else:
            tokenized_text = self.tokenizer.tokenize(
                np.random.choice(self._text_annos[idx].split('\t')),
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
        return dict(**preprocess(video, self.resolution), text=tokenized_text)


class HDF5Dataset_vtokens(data.Dataset):
    """ Dataset for video tokens stored in h5py as int64 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """

    def __init__(self, data_file, sequence_length, train=True, resolution=15, spatial_length=15, image_channels=3,
                 sample_every_n_frames=1):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.image_channels = image_channels
        self.spatial_length = spatial_length

        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._tokens = np.array(self.data[f'{self.prefix}_data'])
        self._idx = np.array(self.data[f'{self.prefix}_idx'][:-1])
        # self._labels = np.array(self.data[f'{self.prefix}_label'])
        self.size = len(self._idx)

        self.sample_every_n_frames = sample_every_n_frames

    @property
    def n_classes(self):
        return np.max(self._labels)+1 if self._labels else 0

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_tokens'] = None
        state['_idx'] = None
        # state['_labels'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._tokens = np.array(self.data[f'{self.prefix}_data'])
        self._idx = np.array(self.data[f'{self.prefix}_idx'][:-1])
        # self._labels = np.array(self.data[f'{self.prefix}_label'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._tokens)
        if end - start <= self.sequence_length:
            return self.__getitem__(torch.randint(low=0, high=self.size, size=(1,)).item())
        # print(end, start, self._idx[idx + 1], self._idx.shape)
        start = start + torch.randint(low=0, high=end - start - self.sequence_length, size=(1,)).item()
        # start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        if self.spatial_length == self.resolution:
            video = torch.tensor(self._tokens[start:start + self.sequence_length]).long()
            box = 0
        else:
            y_start = torch.randint(low=0, high=self.resolution-self.spatial_length+1, size=(1,)).item()
            y_end = y_start + self.spatial_length
            x_start = torch.randint(low=0, high=self.resolution-self.spatial_length+1, size=(1,)).item()
            x_end = x_start + self.spatial_length
            video = torch.tensor(self._tokens[start:start + self.sequence_length, y_start:y_end, x_start:x_end]).long()
            box = np.array([y_start, y_end, x_start, x_end])
            # print(self._tokens.shape, video.shape)
        # skip frames
        if self.sample_every_n_frames > 1:
            video = video[::self.sample_every_n_frames]
        # label = self._labels[idx]
        return dict(video=video, cbox=box)
        # return dict(video=video, label=label, cbox=box)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def preprocess_image(image):
    # [0, 1] => [-1, 1]
    img = image - 0.5
    img = torch.from_numpy(img)
    return img


class FrameDataset(data.Dataset):
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0], meta[2])
            frames = [
                os.path.join(root, item) for item in frames
                if is_image_file(item)
            ]
            if len(frames) > max(0, self.sequence_length * self.sample_every_n_frames):
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all

    def __init__(self, data_folder, sequence_length, resolution=64, sample_every_n_frames=1, train=True):
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.sample_every_n_frames = sample_every_n_frames
        self.data_all = self.load_video_frames(data_folder)

    def __getitem__(self, index):
        batch_data = self.getTensor(index)
        return_list = {'video': batch_data}

        return return_list

    def getTensor(self, index):
        video = self.data_all[index]
        video_len = len(video)

        # load the entire video when sequence_length = -1, whiel the sample_every_n_frames has to be 1
        if self.sequence_length == -1:
            assert self.sample_every_n_frames == 1
            start_idx = 0
            end_idx = video_len
        else:
            n_frames_interval = self.sequence_length * self.sample_every_n_frames
            start_idx = random.randint(0, video_len - 1 - n_frames_interval)
            end_idx = start_idx + n_frames_interval
        img = Image.open(video[0])
        h, w = img.height, img.width

        if h > w:
            half = (h - w) // 2
            cropsize = (0, half, w, half + w)  # left, upper, right, lower
        elif w > h:
            half = (w - h) // 2
            cropsize = (half, 0, half + h, h)

        images = []
        for i in range(start_idx, end_idx,
                       self.sample_every_n_frames):
            path = video[i]
            img = Image.open(path)

            if h != w:
                img = img.crop(cropsize)

            img = img.resize(
                (self.resolution, self.resolution),
                Image.ANTIALIAS)
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img_tensor = preprocess_image(img).unsqueeze(0)
            images.append(img_tensor)

        video_clip = torch.cat(images).permute(3, 0, 1, 2)
        return video_clip

    def __len__(self):
        return self.video_num




class StftDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """

    def __init__(self, data_folder, sequence_length, train=True, resolution=96, sample_every_n_frames=1):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.sample_every_n_frames = sample_every_n_frames
        self.resolution = resolution
        self.exts = ["pickle"]
        self.load_vid_len = 90

        folder = osp.join(data_folder, 'train' if train else 'test')
        self.stft_paths = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])
        self.video_paths = [path.replace("/stft/", "/video/").replace(".pickle", ".mp4") for path in self.stft_paths]

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{self.load_vid_len}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(self.video_paths, self.load_vid_len, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(self.video_paths, self.load_vid_len,
                               _precomputed_metadata=metadata)

        # self._clips = clips.subset(np.arange(24))
        self._clips = clips
        self.n_classes = 0
        # print(len(self._clips), self._clips.num_clips(), len(self.stft_paths), len(self.video_paths))
        # assert self._clips.num_clips()==len(self.stft_paths)

    def __len__(self):
        return len(self.stft_paths)

    def __getitem__(self, idx):
        while True:
            try:
                video, _, _, video_idx = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break

        start = int(random.random() * (self.load_vid_len - self.sequence_length + 1))
        end = start + self.sequence_length

        stft_pickle = self.stft_paths[video_idx]
        with open(stft_pickle, "rb") as f:
            stft = pickle.load(f)
        stft = stft[start:end].astype(np.float32)
        stft = (torch.tensor(stft) * 2 - 1).unsqueeze(0)
        stft = F.interpolate(stft, size=(64, 16), mode="bilinear", align_corners=False)

        return dict(**preprocess(video[start:end], self.resolution, sample_every_n_frames=self.sample_every_n_frames), 
                    stft=stft, path=self.video_paths[video_idx])
