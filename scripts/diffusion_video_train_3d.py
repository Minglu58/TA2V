"""
Train a diffusion model on videos.
"""

import argparse

from diffusion import dist_util, logger
from diffusion.image_datasets import load_data
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.train_util import TrainLoop
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tav import VideoData


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = VideoData(args)
    data = data.train_dataloader()
    #data = load_data(data_dir=args.data_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond,)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        #first_stage_key=args.first_stage_key,
        #cond1_stage_key=args.cond1_stage_key,
        #cond2_stage_key=args.cond2_stage_key,
        text_emb_model = args.text_emb_model,
        save_dir = args.save_dir
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        #batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_channels=128,
        #vtokens_pos=False,    
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    
    #parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
    #parser.add_argument('--cond1_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
    #parser.add_argument('--cond2_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
    parser.add_argument('--save_dir', type=str)
    
    add_dict_to_argparser(parser, defaults)
    
    return parser


if __name__ == "__main__":
    main()
