# TA2V: Text-Audio Guided Video Generation
This is the official implement of our proposed Text&Audio-guided Video Maker (TAgVM) model of TA2V task. Since we pay more attention to music performance video generation, given both the text prompt and the audio signals as input, the model is able to synthesize motion or gesture of the players moving with corresponding melody and rhythm.

<img width="800" alt="our TgVG-TAgVM model" src="https://github.com/Minglu58/TA2V/assets/95404453/12ff6304-c7cd-4eda-a3bb-a07949a0a859">

## Examples
### Music Performance Videos

![generation_stage2_5_db_39_Jerusalem_26_28](https://github.com/Minglu58/TA2V/assets/95404453/4fa0bf08-1ef4-4011-b8e1-5e519fd811d7)
![1_vn_44_K515_15_5](https://github.com/Minglu58/TA2V/assets/95404453/f2d520e8-c544-4a4b-bb18-3313a689d280)
![2_tpt_42_Arioso_79_12](https://github.com/Minglu58/TA2V/assets/95404453/32fd7a40-9c4d-4329-9b20-7946c116b298)
![1_fl_40_Miserere_13_47](https://github.com/Minglu58/TA2V/assets/95404453/9893d660-7656-4440-9528-679f80b3d468)

### Landscape Videos
![fire_crackling_136_6_34](https://github.com/Minglu58/TA2V/assets/95404453/36dadd13-f736-41da-a7c0-c936e45d8ec6)
![fire_crackling_141_2_1](https://github.com/Minglu58/TA2V/assets/95404453/1420c2d8-269a-4f78-b11a-c3bdfd00558a)
![splashing_water_143_6_17](https://github.com/Minglu58/TA2V/assets/95404453/767bdc3f-a86a-4ded-976e-7caecfaf6259)
![squishing_water_136_8_38](https://github.com/Minglu58/TA2V/assets/95404453/281cf914-709f-4d58-85db-56c6f8f49918)


### Failure
![underwater_bubbling_119_7_21](https://github.com/Minglu58/TA2V/assets/95404453/8b199e48-4fd4-4df5-aff3-815f56366bdb)
![raining_145_3_37](https://github.com/Minglu58/TA2V/assets/95404453/319a139d-3be3-4de7-b634-e2c065b06f53)


## Setup
1. Create the virtual environment
```bash
conda create -n tav python==3.9
conda activate tav
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning==1.5.4 einops ftfy h5py imageio regex scikit-image scikit-video tqdm lpips blobfile mpi4py opencv-python-headless kornia termcolor pytorch-ignite visdom piq joblib av==10.0.0 matplotlib ffmpeg==4.2.2 pillow==9.5.0
pip install git+https://github.com/openai/CLIP.git wav2clip transformers
```
2. Create a `saved_ckpts` folder to download pretrained checkpoints.

## Datasets
We create two three-modality datasets named as [URMP-VAT] and [Landscape-VAT](https://drive.google.com/file/d/1dzUa702fN8KaIv71ctuJM0t5sjTLvjjX/view?usp=drive_link), where there are four folders (mp4, stft_pickle, audio, txt) in each training dataset or testing dataset.

You can download these processed datasets to `datasets` folder.
```
URMP-VAT (Landscape-VAT)
  |---train
    |---mp4
    |---stft_pickle
    |---txt
    |---audio
  |---test
    |---mp4
    |---stft_pickle
    |---txt
    |---audio
```
## Download pre-trained checkpoints
| Dataset | VQGAN | GPT |  Diffusion |
| --------------- | --------------- | --------------- | --------------- |
| URMP-VAT | [URMP-VAT_video_VQGAN.ckpt](https://drive.google.com/file/d/1y3n56QWhQLi8fqV9VG0SkpiCQ_VxTIPS/view?usp=drive_link) | [URMP-VAT_GPT.ckpt](https://drive.google.com/file/d/1T1fxotvq6s1Ljdhv3-G4SUsJwpX1gIY4/view?usp=drive_link) | [URMP-VAT_diffusion.pt](https://drive.google.com/file/d/1wmfWX0sF-1TTUxGN5X5088eNXCB3naz1/view?usp=drive_link)
| Landscape-VAT | [Landscape-VAT_video_VQGAN.ckpt](https://drive.google.com/file/d/1MyLU7kRUKpRpyYxnhmpwhbjq5w9RhTyg/view?usp=drive_link) | [Landscape-VAT_GPT.ckpt](https://drive.google.com/file/d/1G_OOT-2oxdGOlLvp7H0RN522SbC9PpVK/view?usp=drive_link) | [Landscape-VAT_diffusion.pt](https://drive.google.com/file/d/1Pc54qlJMeLzB8yE_8pnCiItvFlwsFmwA/view?usp=drive_link)

Since we utilize [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) model to encode audio and images, you can download the checkpoints in their open project page.

## Sampling Procedure
### Sample Short Music Performance Videos
- `gpt_text_ckpt`: path to GPT checkpoint
- `vqgan_ckpt`: path to video VQGAN checkpoint
- `data_path`: path to dataset, you can change it to `post_landscape` for Landscape-VAT dataset
- `load_vid_len`: for URMP-VAT, it is set to `90` (fps=30); for Landscape-VAT, it is set to `30` (fps=10)
- `text_emb_model`: model to encode text, choices: `bert`, `clip`
- `audio_emb_model`: model to encode audio, choices: `audioclip`, `wav2clip`
- `text_stft_cond`: load text-audio-video data
- `n_sample`: the number of videos need to be sampled
- `run`: index for each run
- `resolution`: resolution used in training video VQGAN procedure
- `model_output_size`: the resolution when training the diffusion model
- `audio_guidance_lambda`: coefficient to control audio guidance
- `direction_lambda`: coefficient to control semantic change consistency of audio and video
- `text_guidance_lambda`: coefficient to control text guidance
- `diffusion_ckpt`: path to diffusion model
```
python scripts/sample_tav.py --gpt_text_ckpt saved_ckpts/best_checkpoint-val_text_loss=2.74.ckpt --text_stft_cond \
--vqgan_ckpt saved_ckpts/epoch=6-step=35999-train_recon_loss=0.15.ckpt --text_emb_model bert \
--data_path datasets/post_URMP/ --top_k 2048 --top_p 0.80 --n_sample 50 --run 17 --dataset URMP --load_vid_len 90 \
--audio_emb_model audioclip --resolution 96 --batch_size 1 --model_output_size 128 --noise_schedule cosine \
--iterations_num 1 --audio_guidance_lambda 10000 --direction_lambda 5000 --text_guidance_lambda 10000 \
--diffusion_ckpt saved_ckpts/model300000.pt
```
### Calculate Evaluation Metrics
- `exp_tag`: name of result folder, which is under `results` folder
- `audio_folder`: audio folder name, default: `audio`
- `video_folder`: video folder name, choices: `fake_stage1`, `fake_stage2`, `real`
- `txt_folder`: text folder name, default: `txt`
* **CLIP audio score**
```
python tools/clip_score/clip_audio.py --exp_tag 1_tav_URMP --audio_folder audio --video_folder fake_stage2 --audio_emb_model audioclip
```
* **CLIP text score**
```
python tools/clip_score/clip_text.py --exp_tag 1_tav_URMP --txt_folder txt --video_folder fake_stage2 --batch_size 5
```

- `real_folder`: ground-truth video folder name, default: `real`
- `fake_folder`: generated video folder name, choices: `fake_stage1`, `fake_stage2`
- `mode`: mode to calculate FVD, FID scores, choices: `full`, `size`
* **FVD**
```
python tools/tf_fvd/fvd.py --exp_tag 1_tav_URMP --real_folder real --fake_folder fake_stage2 --mode full
```
* **FID**
```
python tools/tf_fvd/fid.py --exp_tag 1_tav_URMP --real_folder real --fake_folder fake_stage2 --mode full
```
## Training Procedure
You can also train the models on customized datasets. Here we provide the command to train VQGAN, Transformer and Diffusion models.
### video VQGAN
- `embedding_dim`: dimension of codebook embeddings, default: `256`
- `n_codes`: size of codebook, default: `16384`
- `n_hiddens`: hidden channels base, default: `32`
- `downsample`: ratio of downsampling, default: `4 8 8`, `4` for temporal dimension and `8 8` for spatial dimension
- `lr`: learning rate
- `data_path`: path to dataset
- `default_root_dir`: path to save checkpoints
- `resolution`: video resolution to train, we set `96` for URMP-VAT, `64` for Landscape-VAT
- `sequence_length`: length of videos, default: `16`
```
python scripts/train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
--gpus 1 --batch_size 8 --num_workers 16 --accumulate_grad_batches 6 --progress_bar_refresh_rate 100 --max_steps 100000 \
--gradient_clip_val 1.0 --lr 6.0e-5 --data_path datasets/post_URMP/ --default_root_dir path/to/save \
--resolution 9 --sequence_length 16 --discriminator_iter_start 10000 --norm_type batch --perceptual_weight 4 \
--image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4
```
### Transformer
- `first_stage_key`: load first stage data from a batch, default: `video`
- `cond1_stage_key`: load condition stage data from a batch, default: `text`
- `vqvae`: path to load pretrained VQGAN checkpoints
- `n_layer`: the number of layers in transformer
- `n_head`: the number of heads in transformer
- `n_embd`: dimension of embeddings in transformer
- `text_seq_len`: maximum length of text tokens
- `embd_pdrop`: dropout ratio in embedding step
- `resid_pdrop`: dropout ratio in transformer blocks
- `attn_pdrop`: dropout ratio in attention blocks
```
python scripts/train_text_transformer.py --num_workers 4 --val_check_interval 0.5 --progress_bar_refresh_rate 100 \
--gpus 1 --sync_batchnorm --batch_size 8 --first_stage_key video --cond1_stage_key text --text_stft_cond --text_emb_model bert \
--vqvae path/to/video/vqgan --data_path datasets/post_URMP/ --load_vid_len 30 --default_root_dir path/to/save \
--base_lr 4.5e-05 --first_stage_vocab_size 16384 --block_size 1024 --n_layer 12 --n_head 8 --n_embd 512 --resolution 96 \
--sequence_length 16 --text_seq_len 12 --max_steps 500000 --embd_pdrop 0.2 --resid_pdrop 0.2 --attn_pdrop 0.2
```
### Diffusion
- `save_dir`: path to save checkpoints
- `diffusion_steps`: the number of steps to denoise
- `noise_schedule`: choices: `cosine`, `linear`
- `num_channels`: latent channels base
- `num_res_blocks`: the number of resnet blocks in diffusion
- `class_cond`: whether using class or not
- `image_size`: resolution of videos/images
```
python scripts/diffusion_video_train_3d.py --num_workers 8 --gpus 1 --batch_size 1 --text_stft_cond --data_path datasets/post_URMP/ \
--load_vid_len 30 --save_dir path/to/save --resolution 128 --sequence_length 16 --diffusion_steps 4000 --noise_schedule cosine \
--lr 5e-5 --num_channels 128 --num_res_blocks 3 --class_cond False  --log_interval 50 --save_interval 5000 --image_size 128 --learn_sigma True
```
We use 3D diffusion here, setting dims=3 in U-Net for convenience.
## Acknowledgements
Our code is based on [TATS](https://github.com/songweige/TATS) and [blended-diffusion](https://github.com/omriav/blended-diffusion).
