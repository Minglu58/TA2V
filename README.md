# TA2V: Text-Audio Guided Video Generation![image](https://github.com/Minglu58/TATS/assets/95404453/83ecacee-d5e5-46e4-ad2a-4f0d1f209bd6)
This is the official implement of our TgVG-TAgVM model of TA2V task. Since we pay more attention to music performance video generation, given both the text prompt and the audio signals as input, the model is able to synthesize motion or gesture of the players moving with corresponding melody and rhythm.
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


## Sampling Procedure

## Training Procedure

## Acknowledgements
Our code is based on [TATS](https://github.com/songweige/TATS) and [blended-diffusion](https://github.com/omriav/blended-diffusion).
