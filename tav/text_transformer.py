# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import clip
from transformers import BertTokenizer

from .utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from .modules.gpt import GPT
from .modules.encoders import Labelator, SOSProvider, Identity

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class TextTransformer(pl.LightningModule):
    def __init__(self,
                 args,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond1_stage_key="text",
                 #cond2_stage_key="stft",
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.args = args
        self.class_cond_dim = args.class_cond_dim
        self.be_unconditional = args.unconditional
        self.text_emb_model = args.text_emb_model
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond1_stage_key = cond1_stage_key
        #self.cond2_stage_key = cond2_stage_key
        self.vtokens = args.vtokens
        self.sample_every_n_latent_frames = getattr(args, 'sample_every_n_latent_frames', 0)
        
        self.init_first_stage_from_ckpt(args)
        self.init_cond1_stage_from_ckpt(args)
        #self.init_cond2_stage_from_ckpt(args)

        #gpt_cond_stage_vocab_size = self.first_stage_vocab_size + self.cond1_stage_vocab_size + self.cond2_stage_vocab_size
        gpt_cond_stage_vocab_size = self.first_stage_vocab_size + self.cond1_stage_vocab_size
        
        self.transformer_text = GPT(args, gpt_cond_stage_vocab_size, args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                                    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop, 
                                    vtokens_pos=args.vtokens_pos, n_unmasked=args.n_unmasked)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        self.save_hyperparameters()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if not args.vtokens:
            self.first_stage_model = load_vqgan(args.vqvae)
            for p in self.first_stage_model.parameters():
                p.requires_grad = False
            self.first_stage_model.codebook._need_init = False
            self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes
        else:
            self.first_stage_model = None
            self.first_stage_vocab_size = 16384
            # self.first_stage_vocab_size = self.args.first_stage_vocab_size

    def init_cond1_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if self.cond1_stage_key =='label' and not self.be_unconditional:
            model = Labelator(n_classes=args.class_cond_dim)
            model = model.eval()
            model.train = disabled_train
            self.cond1_stage_model = model
            self.cond1_stage_vocab_size = self.class_cond_dim
        elif self.cond1_stage_key =='stft':
            self.cond1_stage_model = load_vqgan(args.stft_vqvae)
            for p in self.cond1_stage_model.parameters():
                p.requires_grad = False
            self.cond1_stage_model.codebook._need_init = False
            self.cond1_stage_model.eval()
            self.cond1_stage_model.train = disabled_train
            self.cond1_stage_vocab_size = self.cond1_stage_model.codebook.n_codes
        elif self.cond1_stage_key =='text':
            if self.text_emb_model == 'clip':
                self.cond1_stage_model, self.preprocess = clip.load("ViT-B/32")
                for p in self.cond1_stage_model.parameters():
                    p.requires_grad = False
                self.cond1_stage_model.eval()
                self.cond1_stage_model.train = disabled_train
                self.cond1_stage_vocab_size = 6244
            elif self.text_emb_model == 'bert':
                self.cond1_stage_model = BertTokenizer.from_pretrained('bert-base-uncased')
                self.cond1_stage_vocab_size = 21128
            else:
                self.cond1_stage_model = Identity()
                self.cond1_stage_vocab_size = 49408
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond1_stage_key = self.first_stage_key
            self.cond1_stage_model = SOSProvider(self.sos_token)
            self.cond1_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond1_stage_key)            
    '''
    def init_cond2_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if self.cond2_stage_key =='stft':
            self.cond2_stage_model = load_vqgan(args.stft_vqvae)
            for p in self.cond2_stage_model.parameters():
                p.requires_grad = False
            self.cond2_stage_model.codebook._need_init = False
            self.cond2_stage_model.eval()
            self.cond2_stage_model.train = disabled_train
            self.cond2_stage_vocab_size = self.cond2_stage_model.codebook.n_codes
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond2_stage_key = self.first_stage_key
            self.cond2_stage_model = SOSProvider(self.sos_token)
            self.cond2_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond2_stage_key)
    '''
    
    def forward(self, x, c1, cbox=None):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x) # torch.Size([4, 576])
        _, c1_indices = self.encode_to_c1(c1) # torch.Size([4, 12])
        #z_indices = z_indices + self.cond1_stage_vocab_size + self.cond2_stage_vocab_size
        z_indices = z_indices + self.cond1_stage_vocab_size

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices
 
        c1z_indices = torch.cat((c1_indices, a_indices), dim=1)  # torch.Size([4,588])

        # target1: 1st frame embeddings
        target1 = z_indices
        
        # make the 1st frame prediction using text
        logits1, _ = self.transformer_text(c1z_indices[:, :-1], cbox=cbox)
        logits1 = logits1[:, c1_indices.shape[1]-1:] #torch.Size([4, 576, _]) 
        
        return logits1, target1

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            # x = vq_output['embeddings'], targets = vq_output['encodings']
            x, targets = self.first_stage_model.encode(x, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                x = x[:, :, ::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            targets = targets.reshape(targets.shape[0], -1)
        return x, targets

    @torch.no_grad()
    def encode_to_c1(self, c):
        if self.text_emb_model == 'clip':
            indices = self.cond1_stage_model.encode_text(c) * 1000.0
            indices = indices - indices.min()
            indices = indices.to(torch.long)
            quant_c = self.cond1_stage_model.encode_text(c)
        elif self.text_emb_model == 'bert':
            quant_c = torch.squeeze(c["input_ids"], 1)
            indices = torch.squeeze(c["input_ids"], 1)
        else:
            quant_c, indices = self.cond1_stage_model.encode(c, include_embeddings=True)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c1 = self.get_input(self.cond1_stage_key, batch)
        if N is not None:
            x = x[:N]
            c1 = c1[:N]
        return x, c1

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c1 = self.get_xc(batch)
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        #print('train:', x.min(), x.max(), x.shape, c1, c2)
        logits1, target1 = self(x, c1, cbox)
        loss1 = F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), target1.reshape(-1))
        #loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), target2.reshape(-1))
        #loss = loss1 + loss2
        text_acc1, text_acc5 = accuracy(logits1.reshape(-1, logits1.shape[-1]), target1.reshape(-1), topk=(1, 5))
        #audio_acc1, audio_acc5 = accuracy(logits2.reshape(-1, logits2.shape[-1]), target2.reshape(-1), topk=(1, 5))
        return loss1, text_acc1, text_acc5

    def training_step(self, batch, batch_idx):
        loss1, text_acc1, text_acc5 = self.shared_step(batch, batch_idx)
        #loss = loss1 + loss2
        self.log("train_text_loss", loss1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_text_acc1', text_acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_text_acc5', text_acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss1

    def validation_step(self, batch, batch_idx):
        loss1, text_acc1, text_acc5 = self.shared_step(batch, batch_idx)
        #loss = loss1 + loss2
        self.log("val_text_loss", loss1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_text_acc1', text_acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_text_acc5', text_acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss1

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer_text.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.args.vtokens_pos:
            no_decay.add('vtokens_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer_text.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.80},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate/2.0, max_lr=self.learning_rate, cycle_momentum=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
        #optimizer = torch.optim.SGD(optim_groups, lr=self.learning_rate, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]
        #return optimizer


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--stft_vqvae', type=str, help='path to vqgan ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        # VideoGPT hyperparmeters
        parser.add_argument('--vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)
        parser.add_argument('--block_size', type=int, default=256)
        parser.add_argument('--n_layer', type=int, default=48)
        parser.add_argument('--n_head', type=int, default=24)
        parser.add_argument('--n_embd', type=int, default=1536)
        parser.add_argument('--embd_pdrop', type=float, default=0)
        parser.add_argument('--resid_pdrop', type=float, default=0)
        parser.add_argument('--attn_pdrop', type=float, default=0)
        parser.add_argument('--n_unmasked', type=int, default=0)
        parser.add_argument('--sample_every_n_latent_frames', type=int, default=0)
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond1_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
        parser.add_argument('--cond2_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])

        return parser

