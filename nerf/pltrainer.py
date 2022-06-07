from typing import Optional

import os
import torch

import imageio
import trimesh
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from torch_ema import ExponentialMovingAverage

from nerf.utils import extract_geometry

class RenderCallback(Callback):
    def __init__(self, dirpath, name):
        self.dirpath = dirpath
        self.name = name
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return 
        
class RenderMeshCallback(Callback):
    def __init__(self, dirpath, name, resolution=256, threshold=10) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.name = name
        self.resolution = resolution
        self.threshold = threshold
        os.makedirs(dirpath, exist_ok=True)

    def save_mesh(self, pl_module, path):
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=pl_module.hparams.fp16):
                    sigma = pl_module.model.density(pts.to(pl_module.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(pl_module.model.aabb_infer[:3], pl_module.model.aabb_infer[3:], resolution=self.resolution, threshold=self.threshold, query_func=query_func)            
        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(path)

    def on_train_end(self, trainer, pl_module):
        path = os.path.join(self.dirpath, f'{self.name}_{pl_module.current_epoch}.ply')
        self.save_mesh(pl_module=pl_module, path=path)

class RenderGifCallback(Callback):
    def __init__(self, dirpath, name, dataloader) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.name = name
        self.dataloader = dataloader

    def save_gif(self, pl_module, path):
        local_step = 0
        if pl_module.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        
        imageio.mimsave(path, imgs, fps=30)
    def on_train_end(self, trainer, pl_module):
        path = os.path.joint(self.dirpath, f'{self.name}_{pl_module.current_epoch}.gif')
        self.save_gif(pl_module=pl_module, path=path)


class NeRFModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, criterion, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.criterion = criterion
        if hparams.ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=hparams.ema_decay)
        else:
            self.ema = None
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.valid_psnr = torchmetrics.PeakSignalNoiseRatio()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def on_train_epoch_start(self) -> None:
        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.hparams.fp16):
                self.model.update_extra_state()

    def on_validation_epoch_start(self) -> None:
        with torch.no_grad():
            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.hparams.fp16):
                    self.model.update_extra_state()

    def training_step(self, data, batch_idx):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]


        images = data['images'] # [B, N, 3/4]
        B, N, C = images.shape
    
        # train in srgb color space
        if C == 4:
            # train with random background color if using alpha mixing
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            bg_color = None
            gt_rgb = images  


        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **vars(self.hparams))
    
        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]                  
        
        # todo: error map

        loss = loss.mean()
        return {'loss': loss, 'pred': pred_rgb, 'gt': gt_rgb,}

    def training_step_end(self, step_output):
        self.log("train/loss", step_output["loss"].detach())
        self.train_psnr(step_output["pred"], step_output["gt"])
        self.log("train/psnr", self.train_psnr, prog_bar=True)

    def on_training_epoch_end(self):
        if self.ema is not None:
            self.ema.update()

        return super().on_training_epoch_end()

    def on_validation_epoch_start(self) -> None:
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_epoch_end(self) -> None:
        if self.ema is not None:
            self.ema.restore()

        return super().on_validation_epoch_end()

    def validation_step(self, data, batch_idx):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.hparams))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        return {'loss': loss, 'pred_rgb': pred_rgb, 'pred_depth': pred_depth, 'gt_rgb': gt_rgb}     

    def validation_step_end(self, step_output):
        self.log("valid/loss", step_output["loss"].detach())
        self.valid_psnr(step_output["pred_rgb"], step_output["gt_rgb"])
        self.log("valid/psnr", self.train_psnr, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(self.model.encoder.parameters())},
            {'name': 'net', 'params': list(self.model.sigma_net.parameters()) + list(self.model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=self.hparams.lr, betas=(0.9, 0.99), eps=1e-15)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / self.hparams.iters, 1))
        return [optimizer], [scheduler]