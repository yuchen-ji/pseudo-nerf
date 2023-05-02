from ast import arg
import math
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
import torch.nn as nn


def inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    # rays_o (N, H, W, 3): ray origins
    # rays_d (N, H, W, 3): ray directions
    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def train_meta(args, meta_model, meta_optim, data_loader, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """
    for imgs, poses, hwf, bound in data_loader:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        meta_optim.zero_grad()

        inner_model = copy.deepcopy(meta_model)
        inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)

        inner_loop(inner_model, inner_optim, imgs, poses,
                    hwf, bound, args.num_samples,
                    args.train_batchsize, args.inner_steps)
        
        with torch.no_grad():
            for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                # grad is the delta between inner loop and outer loop(which before m step)
                meta_param.grad = meta_param - inner_param
        
        meta_optim.step()


def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def val_meta(args, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    
    val_psnrs = []
    for imgs, poses, hwf, bound in val_loader:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        val_model.load_state_dict(meta_trained_state)
        val_optim = torch.optim.SGD(val_model.parameters(), args.tto_lr)

        # pose is in the val-set, use this pose to calculate loss
        inner_loop(val_model, val_optim, tto_imgs, tto_poses, hwf,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps)
        
        scene_psnr = report_result(val_model, test_imgs, test_poses, hwf, bound, 
                                    args.num_samples, args.test_batchsize)
        val_psnrs.append(scene_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, default="./configs/shapenet/chairs.json",
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--gpu_id', type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
    
    # different model use different data, prevent cpu跑满，导致GPU效率低
    args.dataset_root="./data/shapenet/chairs_{}".format(args.gpu_id+1)

    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # init weight and bias
    meta_model = build_nerf(args)
    for m in meta_model.net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(10))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias.data, -bound, bound)

    meta_model.to(device)
    # outer loop learning rate
    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)

    for epoch in range(1, args.meta_epochs+1):
        train_meta(args, meta_model, meta_optim, train_loader, device)
        val_psnr = val_meta(args, meta_model, val_loader, device)
        print(f"Epoch: {epoch}, val psnr: {val_psnr:0.3f}")

        torch.save({
            'epoch': epoch,
            'meta_model_state_dict': meta_model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, 
            f'./model_weight/model_{args.gpu_id+1}/model_{args.gpu_id+1}_meta_epoch{epoch}.pth')


if __name__ == '__main__':
    main()