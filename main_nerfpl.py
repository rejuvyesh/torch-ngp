import torch
import argparse
import os.path

from nerf.provider import NeRFDataset
from nerf.pltrainer import NeRFModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    parser.add_argument('--xyzencoder', type=str, default='hashgrid', help="specify encoder type")

    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.ff:
        opt.fp16 = True
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding=opt.xyzencoder,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=10 if opt.mode == 'blender' else 1,
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = NeRFDataset(opt, device=device, type='train').dataloader()
    valid_dataloader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
    mod = NeRFModel(model, criterion, opt)

    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(opt.workspace, "checkpoints"), filename='{epoch:d}.pth.tar', monitor='valid/psnr', mode='max', save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]    
    logger = TensorBoardLogger(save_dir=os.path.join(opt.workspace, "logs"),
                               name="ngp",
                               default_hp_metric=False)
    trainer = pl.Trainer(default_root_dir=opt.workspace, max_steps=opt.iters, logger=logger,  callbacks=callbacks,
                        accelerator='gpu', devices=1, strategy=None, precision=16 if opt.fp16 else 32)
    trainer.fit(mod, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()