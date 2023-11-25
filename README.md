# TA2V: Text-Audio Guided Video Generation
This is the official implement of our proposed TgVG-TAgVM model of TA2V task. Since we pay more attention to music performance video generation, given both the text prompt and the audio signals as input, the model is able to synthesize motion or gesture of the players moving with corresponding melody and rhythm.

<img width="800" alt="our TgVG-TAgVM model" src="https://github.com/Minglu58/TA2V/assets/95404453/12ff6304-c7cd-4eda-a3bb-a07949a0a859">

## Examples

## Setup
1. Create the virtual environment
```bash
conda create -n tav python==3.9
conda activate tav
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning==1.5.4 einops ftfy h5py imageio regex scikit-image scikit-video tqdm lpips blobfile mpi4py opencv-python-headless kornia termcolor pytorch-ignite visdom piq joblib av==10.0.0 matplotlib ffmpeg==4.2.2 pillow==9.5.0
pip install git+https://github.com/openai/CLIP.git wav2clip transformers
```
2. Create a `saved_ckpts` folder to download pretrained checkpoints [Here]().

## Datasets
We create two three-modality datasets named as [URMP-VAT]() and [Landscape-VAT](), where there are four folders (mp4, stft_pickle, audio, txt) in each training dataset or testing dataset.

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
| URMP-VAT | [epoch]() | [epoch]() | [model.pt]()
| Landscape-VAT | [epoch]() | [epoch]() | [model.pt]()

## Sampling Procedure
You can sample the short music performance videos using following command.

- `gpt_text_ckpt`: the path to gpt checkpoint
```
python scripts/sample_tav.py --gpt_text_ckpt saved_ckpts/best_checkpoint-val_text_loss=2.74.ckpt --text_stft_cond \
--vqgan_ckpt saved_ckpts/epoch=6-step=35999-train_recon_loss=0.15.ckpt --text_emb_model bert \
--data_path datasets/post_URMP/ --top_k 2048 --top_p 0.80 --n_sample 50 --run 17 --dataset URMP --load_vid_len 90 \
--audio_emb_model audioclip --resolution 96 --batch_size 1 --model_output_size 128 --noise_schedule cosine \
--iterations_num 1 --audio_guidance_lambda 10000 --direction_lambda 5000 --text_guidance_lambda 0 \
--diffusion_ckpt saved_ckpts/model300000.pt
```
To compute the evaluation metrics, run
- CLIP audio score
```
python tools/clip_score/clip_audio.py --exp_tag 1_tav_URMP --audio_folder audio --video_folder fake_stage2 --audio_emb_model audioclip
```
- CLIP text score
```
python TAV/tools/clip_score/clip_text.py --exp_tag 1_tav_URMP --txt_folder txt --video_folder fake_stage2 --batch_size 5
```
- FVD
```
python tools/tf_fvd/fvd.py --exp_tag 1_tav_URMP --real_folder real --fake_folder fake_stage2 --mode full
```
- FID
```
python tools/tf_fvd/fid.py --exp_tag 1_tav_URMP --real_folder real --fake_folder fake_stage2 --mode full
```
## Training Procedure

## Acknowledgements
Our code is based on [TATS](https://github.com/songweige/TATS) and [blended-diffusion](https://github.com/omriav/blended-diffusion).
