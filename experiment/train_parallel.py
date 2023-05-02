import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
sys.path.insert(0, '/home/shenxi/interns/JiYuchen/semi-nerf/')

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
import threading


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
    num_ensemble = 8
    print("cuda count:", torch.cuda.device_count())

    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, default="./configs/shapenet/chairs.json",
                        help='config file for the shape class (cars, chairs or lamps)')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device_ensemble = []
    for i in range(num_ensemble):
        device = torch.device("cuda:{}".format(i) if torch.cuda.is_available() else "cpu")
        device_ensemble.append(device)


    # create a loader ensemble & create a model ensemble
    train_loader_ensemble = []
    val_loader_ensemble = []
    model_ensemble = []
    optim_ensemble = []

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                            splits_path=args.splits_path, num_views=args.train_views)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                    splits_path=args.splits_path,
                    num_views=args.tto_views+args.test_views)

    for i in range(num_ensemble):
        # ===========================================================================================

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)

        train_loader_ensemble.append(train_loader)
        val_loader_ensemble.append(val_loader)

        # ===========================================================================================
        meta_model = build_nerf(args)
        meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
        # init random weight & bias
        for m in meta_model.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)
        model_ensemble.append(meta_model)
        optim_ensemble.append(meta_optim)
        # move each model to cuda
        meta_model.to(device_ensemble[i])


    thread1 = creatThread(1, 8, model_ensemble[0], optim_ensemble[0], train_loader_ensemble[0], val_loader_ensemble[0], device_ensemble[0], args)
    thread2 = creatThread(2, 8, model_ensemble[1], optim_ensemble[1], train_loader_ensemble[1], val_loader_ensemble[1], device_ensemble[1], args)
    thread3 = creatThread(3, 8, model_ensemble[2], optim_ensemble[2], train_loader_ensemble[2], val_loader_ensemble[2], device_ensemble[2], args)
    thread4 = creatThread(4, 8, model_ensemble[3], optim_ensemble[3], train_loader_ensemble[3], val_loader_ensemble[3], device_ensemble[3], args)
    thread5 = creatThread(5, 8, model_ensemble[4], optim_ensemble[4], train_loader_ensemble[4], val_loader_ensemble[4], device_ensemble[4], args)
    thread6 = creatThread(6, 8, model_ensemble[5], optim_ensemble[5], train_loader_ensemble[5], val_loader_ensemble[5], device_ensemble[5], args)
    thread7 = creatThread(7, 8, model_ensemble[6], optim_ensemble[6], train_loader_ensemble[6], val_loader_ensemble[6], device_ensemble[6], args)
    thread8 = creatThread(8, 8, model_ensemble[7], optim_ensemble[7], train_loader_ensemble[7], val_loader_ensemble[7], device_ensemble[7], args)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    

def train_each_model(threadID, meta_model, meta_optim, train_loader, val_loader, device, args):
    for epoch in range(1, args.meta_epochs+1):
        train_meta(args, meta_model, meta_optim, train_loader, device)
        val_psnr = val_meta(args, meta_model, val_loader, device)
        print(f"Epoch: {epoch}, val psnr: {val_psnr:0.3f}")

        torch.save({
            'epoch': epoch,
            'meta_model_state_dict': meta_model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, 
            f'./model_weights/model_{threadID}/meta_epoch{epoch}.pth')


class creatThread(threading.Thread):
    def __init__(self, threadID, counter, 
                meta_model, meta_optim, train_loader, val_loader, device, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.counter = counter

        # here are something for training
        self.meta_model = meta_model
        self.meta_optim = meta_optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args


    def run(self):
        print("Start training Model_{}/{}\n".format(self.threadID, self.counter))
        train_each_model(self.threadID, 
                         self.meta_model, 
                         self.meta_optim, 
                         self.train_loader, 
                         self.val_loader, 
                         self.device,
                         self.args)
        print("Exit training Model_{}/{}\n".format(self.threadID, self.counter))


if __name__ == '__main__':
    main()