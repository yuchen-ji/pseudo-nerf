import os, sys
from select import select

from numpy import source
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
sys.path.insert(0, '/home/shenxi/interns/JiYuchen/semi-nerf/')

from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from utils.select_pose import get_delta_pose_all
from models.rendering import get_rays_shapenet, sample_points, volume_render, volume_render_enesmble
import copy
import math
import imageio
import threading
import random
import lpips
import skimage
import numpy as np


def test_time_optimize_uncertainty(args, idx, model, optim, imgs, poses, source_imgs_num, uncertaintys, hwf, bound, inner_steps, device):
    """
    test-time-optimize the meta trained model on available views
    """
    sample_random = False

    num_imgs = len(imgs)
    all_imgs = torch.cat(imgs).to(device)
    all_poses = torch.cat(poses).to(device)
    pseudo_uncertaintys = torch.cat(uncertaintys).to(device) if uncertaintys is not None else None

    ray_origins, ray_directions = get_rays_shapenet(hwf, all_poses)
    rays_o, rays_d = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
    all_pixels_gt = all_imgs.reshape(-1, 3)

    # the labeled imgs's all pixels
    source_pixels_num = source_imgs_num*128*128
    # don't sample random pixels, use the fixed pixels
    indices_range = torch.load('./sampling_pixels/1v_64p/instance_{}.pt'.format(idx+1))

    # the pseudo_imgs's trusted pixels
    if pseudo_uncertaintys is not None:
        # num_pixels'shape is NV*128*128, which is the number of rays
        num_pixels = pseudo_uncertaintys.reshape(-1)
        # the pseudo_pixels that can be trusted
        trusted_pixels = torch.where(num_pixels < 0.1)[0].cpu() + source_pixels_num

    for step in range(inner_steps):
        if sample_random:
            # sample labeled img's all pixels
            indices_source = torch.randint(source_pixels_num, size=[source_imgs_num * args.tto_batchsize])
        else:
            indices_source = indices_range[step]
        # sample pixels from pseudo img's trusted_pixels
        if pseudo_uncertaintys is not None:
            # if the confidence pixel is less than the pixel defined to be sampled, then just sample all confidence pixels
            if len(trusted_pixels) > (num_imgs-source_imgs_num) * args.tto_batchsize:
                selected_index = torch.LongTensor(random.sample(range(len(trusted_pixels)), (num_imgs-source_imgs_num) * args.tto_batchsize))
                indices = torch.index_select(trusted_pixels, 0, selected_index)
                indices = torch.cat((indices_source, indices))
            else:
                indices = torch.cat((indices_source, trusted_pixels))
        else:
            indices = indices_source

        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        # pixelbatch is the sample_pixels in one image
        pixelbatch = all_pixels_gt[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)

        # soft loss weight
        weight_source = torch.tensor([[1,1,1]]).expand(
            source_imgs_num * args.tto_batchsize, 3
        ).float().to(device)
        weight_pseudo = torch.tensor([[0.5,0.5,0.5]]).expand(
            indices.shape[0] - args.tto_batchsize, 3
        ).float().to(device)
        weight = torch.cat((weight_source, weight_pseudo), dim=0)
        
        # ouput & loss
        colors, _ = volume_render_enesmble(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(weight*colors, weight*pixelbatch)
        loss.backward()
        optim.step()


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    lpips_vgg = lpips.LPIPS(net="vgg").to(imgs.device)

    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)
    
    view_lpips = []
    view_psnrs = []
    view_ssims = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    args.num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            # psnr
            error = F.mse_loss(img, synth)
            psnr_ = -10*torch.log10(error)
            # lpips
            lpips_ = lpips_vgg(img.permute(2,0,1), synth.permute(2,0,1))
            # ssim
            ssim_ = skimage.metrics.structural_similarity(img.cpu().numpy(), synth.cpu().numpy(), multichannel=True, data_range=1)
            view_psnrs.append(psnr_)
            view_lpips.append(lpips_)
            view_ssims.append(ssim_)


    scene_psnr = torch.stack(view_psnrs).mean()
    scene_lpips = torch.stack(view_lpips).mean()
    scene_ssim = np.mean(view_ssims)

    return scene_psnr, scene_lpips, scene_ssim


def generate_pseduo_imgs(args, model, pseudo_poses, hwf, bound):

    pseudo_poses = torch.cat(pseudo_poses)
    pseudo_img = []     #use this to store pseudo_imgs
    pseudo_density = []     #use this to store pseudo_density values
    ray_origins, ray_directions = get_rays_shapenet(hwf, pseudo_poses)

    for rays_o, rays_d in zip(ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1], args.num_samples, perturb=False)
        synth = []
        denth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch, density_batch = volume_render_enesmble(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
                denth.append(density_batch)
            synth = torch.cat(synth, dim=0).reshape((128,128,3))
            denth = torch.cat(denth, dim=0).reshape((128,128,1))
            pseudo_img.append(synth)
            pseudo_density.append(denth)

    return pseudo_img, pseudo_density


def calculate_uncertainty():

    global pseudo_imgs_ensemble
    global pseudo_density_ensemble
    pseudo_imgs, uncentaintys = [], []

    # M model & N pose imgs
    # convert from [model_1:[pose_1,pose_2,...,pose_n], ..., model_m:[pose_1,pose2,...,pose_n]]
    # to [pose_1:[model_1,model_2, ..., model_m], ..., pose_n:[model1,...,model_m]]
    pseudo_imgs_ensemble = list(map(list, zip(*pseudo_imgs_ensemble)))
    pseudo_density_ensemble = list(map(list, zip(*pseudo_density_ensemble)))

    # how many imgs in differnt poses loop
    idx = 0
    for imgs, densitys in zip(pseudo_imgs_ensemble, pseudo_density_ensemble):
        idx += 1
        # cat all imgs for different model in one pose
        cat_R, cat_G, cat_B, cat_density = [], [], [], []
        for img, density in zip(imgs, densitys):
            cat_R.append(img[...,0].unsqueeze(2).to("cuda:0"))
            cat_G.append(img[...,1].unsqueeze(2).to("cuda:0"))
            cat_B.append(img[...,2].unsqueeze(2).to("cuda:0"))
            cat_density.append(density.to("cuda:0"))
        
        cat_R, cat_G, cat_B = torch.cat(cat_R, dim=2), torch.cat(cat_G, dim=2), torch.cat(cat_B, dim=2)
        cat_density = torch.cat(cat_density, dim=2)

        mean_R = torch.mean(cat_R, dim=2, keepdims=True)
        mean_G = torch.mean(cat_G, dim=2, keepdims=True)
        mean_B = torch.mean(cat_B, dim=2, keepdims=True)
        mean_density = torch.mean(cat_density, dim=2, keepdim=True)

        std_R = torch.var(cat_R, dim=2, keepdims=True)
        std_G = torch.var(cat_G, dim=2, keepdims=True)
        std_B = torch.var(cat_B, dim=2, keepdims=True)
        std_img = torch.mean(torch.cat((std_R, std_G, std_B), dim=2), dim=2, keepdims=True)
        std_density = torch.square(1 - mean_density)
        
        # print(std_img[50:70,50:70,:])
        # print(std_density[50:70,50:70,:])
        # here we get uncentainty of one img
        uncentainty = std_img*100 + std_density

        # one img's mean RGB from differnt model
        pseudo_img = torch.cat((mean_R, mean_G, mean_B), dim=2)
        # store all pose imgs & all pose's uncertainty
        pseudo_imgs.append(pseudo_img.unsqueeze(0).cpu())
        uncentaintys.append(uncentainty.unsqueeze(0).cpu())

        # torch.save(pseudo_img.cpu().numpy(), './visiualization/tensor/img_{}.pt'.format(idx))
        # torch.save(uncentainty.cpu().numpy(), './visiualization/tensor/img_{}_uncertainty.pt'.format(idx))
        # torch.save(std_img.cpu().numpy(), './visiualization/tensor/img_{}_rgb.pt'.format(idx))
        # torch.save(std_density.cpu().numpy(), './visiualization/tensor/img_{}_density.pt'.format(idx))
    
    # clear global variables: pseudo_imgs_ensemble & uncentaintys_ensemble for store next epoch's pseudo_imgs & uncentaintys
    pseudo_imgs_ensemble.clear()
    pseudo_density_ensemble.clear()

    return pseudo_imgs, uncentaintys


class ensembleThread(threading.Thread):
    def __init__(self, threadID, counter, 
                idx, temp_model, temp_optim, device, 
                source_imgs_num, pseudo_poses,
                mixed_imgs, mixed_poses, uncentaintys,
                hwf, bound, inner_steps, 
                args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.counter = counter

        # here are something for training
        self.idx = idx
        self.temp_model = temp_model
        self.temp_optim = temp_optim
        self.device = device
        self.source_imgs_num = source_imgs_num        
        self.pseudo_poses = pseudo_poses
        self.mixed_imgs = mixed_imgs
        self.mixed_poses = mixed_poses
        self.uncentaintys = uncentaintys
        self.hwf = hwf.to(device)
        self.bound = bound.to(device)
        self.inner_steps = inner_steps
        self.args = args

    def run(self):
        print("Start training model_{}/{}".format(self.threadID, self.counter))
        test_time_optimize_uncertainty(self.args, self.idx, self.temp_model, self.temp_optim, self.mixed_imgs, self.mixed_poses, 
                                        self.source_imgs_num, self.uncentaintys, self.hwf, self.bound, self.inner_steps, self.device)
        # get pseudo_poses according to source_poses
        pseudo_poses = [pose.to(self.device) for pose in self.pseudo_poses]
        # generate_pseduo_imgs
        pseudo_img, pseudo_density = generate_pseduo_imgs(self.args, self.temp_model, pseudo_poses, self.hwf, self.bound)
        # update pseudo_imgs to pseudo_imgs_ensemble
        if lock.acquire():
            pseudo_imgs_ensemble.append(pseudo_img)
            pseudo_density_ensemble.append(pseudo_density)
            lock.release()
        print("Exit training model_{}/{}".format(self.threadID, self.counter))


def test():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, default='./configs/shapenet/chairs.json',
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, default='./model_weight',
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    # =========== training parameters ===========
    num_ensemble = 4
    outer_epoch = 5
    pseudo_poses_num = 3
    pseudo_poses_range = math.pi/2
    # ===========================================

    # get device list, different model use different gpu
    device_ensemble = [torch.device("cuda:{}".format(i) if torch.cuda.is_available() else "cpu") for i in range(num_ensemble)]

    # dataset & dataloader
    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root, splits_path=args.splits_path, num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # get model with different weights
    model_ensemble = []
    for i in range(num_ensemble):
        model = build_nerf(args)
        weight_path = os.path.join(args.weight_path, "model_{}".format(i+1), "model_{}_meta_epoch15.pth".format(i+1))
        checkpoint = torch.load(weight_path, map_location=device_ensemble[i])
        meta_state = checkpoint['meta_model_state_dict']
        model.to(device_ensemble[i])
        model.load_state_dict(meta_state)
        model_ensemble.append(model)

    # savedir
    savedir = Path("./videos_ensemble_4v")
    savedir.mkdir(exist_ok=True)

    test_psnr  = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        print(f"========= Scene {idx+1} =========")
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
        
        # tto_imgs used for training, test for calculate psnr
        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        # split source to list
        source_imgs = list(torch.split(tto_imgs, 1, dim=0))   # labeled imgs: [img1, img2, ..., (1,128,128,3)]
        source_poses = list(torch.split(tto_poses, 1, dim=0))    # labeled poses: [pose1, pose2, ..., (1, 4, 4)]
        mixed_imgs = copy.deepcopy(source_imgs)     # use these pseudo & labeled images to fine_tune!
        mixed_poses = copy.deepcopy(source_poses)   # use these pseudo & labeled poses to fine_tune!
        inner_steps = [1000]    # use these steps to fine_tune each epoch with different pseudo images

        # ==================== outer-loop generator pseudo imgs =================
        for epoch_id in range(outer_epoch):
            print("Epoch: {}/{}".format(epoch_id+1, outer_epoch))
            # in epoch 1, only labeles imgs are used, so the uncentaintys are zeros
            if(epoch_id+1 == 1):
                uncentaintys = None
                # uncentaintys = torch.split(
                #     torch.zeros_like(
                #     torch.cat(source_imgs, dim=0)[...,0]
                #     ).unsqueeze(-1), 1, dim=0  
                # )

            # copy the init model
            temp_model_ensemble = copy.deepcopy(model_ensemble)
            # defined different optimizer
            optim_ensemble = [torch.optim.SGD(temp_model_ensemble[i].parameters(), args.tto_lr) for i in range(num_ensemble)]
            pseudo_poses = get_delta_pose_all(source_poses, pseudo_poses_num, pseudo_poses_range)
            # ==================== here we use multi-thread to train model_ensemble ====================
            threads = [
                ensembleThread(i+1, num_ensemble, 
                    idx, temp_model_ensemble[i], optim_ensemble[i], device_ensemble[i],
                    len(source_imgs), pseudo_poses,
                    mixed_imgs, mixed_poses, uncentaintys,
                    hwf, bound, inner_steps[0],
                    args)
                for i in range(num_ensemble)
            ]
            # start & wait for all threads to finish
            [t.start() for t in threads]
            [t.join() for t in threads]
            # get mean-imgs & uncertainty 
            pseudo_imgs, uncentaintys = calculate_uncertainty()
            # print(pseudo_imgs[0])
            # print(uncentaintys[0])

            mixed_imgs = copy.deepcopy(source_imgs)
            mixed_poses = copy.deepcopy(source_poses)
            mixed_imgs.extend(pseudo_imgs)
            mixed_poses.extend(pseudo_poses)

            # copy the trained model ensemble
            final_model = copy.deepcopy(temp_model_ensemble[0]).to("cuda:0")
            hwf, bound, test_imgs, test_poses = hwf.to("cuda:0"), bound.to("cuda:0"), test_imgs.to("cuda:0"), test_poses.to("cuda:0")     
            
            scene_psnr, scene_lpips, scene_ssim = report_result(args, final_model, test_imgs, test_poses, hwf, bound)
            print(f"scene {idx+1}, psnr: {scene_psnr:.3f}")   
            print(f"scene {idx+1}, lpips: {scene_lpips:.3f}")   
            print(f"scene {idx+1}, ssim: {scene_ssim:.3f}")   


        # create_360_video(args, final_model, hwf, bound, "cuda:0", idx+1, savedir, suffix='ensemble_1v')
        # print("video created!")
        # write psnr to file
        # with open('./video_ensemble_4v/psnr.txt','a') as f:
            # f.write("video_{}: {}\n".format(idx+1, scene_psnr.cpu().numpy()))


if __name__ == '__main__':

    # lock pseudo_dict
    lock = threading.Lock()
    pseudo_imgs_ensemble = []
    pseudo_density_ensemble = []

    test()