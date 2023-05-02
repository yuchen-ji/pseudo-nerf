from marshal import load
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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
from models.rendering import get_rays_shapenet, sample_points, volume_render
import copy
import math
import imageio
import numpy as np


def test_time_optimize(idx, p_idx, args, model, optim, imgs, poses, num_source_imgs, hwf, bound, inner_steps, store_load=False):
    """
    test-time-optimize the meta trained model on available views
    """
    sample_random = False
    num_known_imgs = num_source_imgs + p_idx - 1
    num_konwn_pixels = num_known_imgs*128*128

    # load the instance's sampling pixels
    indices_range = []
    indices_range.append(torch.load('./sampling_pixels/1v_64p/instance_{}.pt'.format(idx+1)))

    # store & load the pseudo's sampling pixels
    if store_load:
        savedir = Path(f"./sampling_pixels/verify/instance_{idx+1}")
        savedir.mkdir(exist_ok=True)

        # 存储新加入训练集的伪标签的采样点， pseudo images' indices need to be sorted
        if p_idx != 0:
            indices_store = []
        # 第一次加入伪标签不用加载他的sampling pixel，而是要存储下来它的sampling pixels
        if p_idx > 1:
            files = os.listdir(savedir)
            files.sort(key = lambda x: int(x[7:-3]))
            for file in files:
                indices_range.append(torch.load(os.path.join(savedir, file)))

    num_imgs = len(imgs)
    all_imgs = torch.cat(imgs)
    all_poses = torch.cat(poses)
    ray_origins, ray_directions = get_rays_shapenet(hwf, all_poses)
    rays_o, rays_d = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
    all_pixels_gt = all_imgs.reshape(-1, 3)
    # num_rays is NV*128*128, which is the number of rays
    num_rays = rays_d.shape[0]

    for step in range(inner_steps):
        # 128*128 rays are too many, so that we should sample a little rays(num = tto_batchsize) for calculate loss

        # sample from the labeled images
        if sample_random:
            indices_source = torch.randint(num_rays, size=[num_imgs * args.tto_batchsize])
        else:
            indices_source = torch.cat(
                [each[step] for each in indices_range]
            )

        # sample from pseudo images & cat sampling pixels
        if num_imgs > num_source_imgs:
            # ******* Notice: 1 * args.tto_batchsize *******
            if store_load:
                indices = torch.randint(num_rays-num_konwn_pixels, size=[1 * args.tto_batchsize])
                # 把每一个step的采样点存储起来
                if p_idx!=0:
                    indices_store.append(indices)
                indices = indices + num_konwn_pixels
            else:
                indices = torch.randint(num_rays-num_source_imgs*128*128, size=[(num_imgs-num_source_imgs) * args.tto_batchsize])
                indices = indices + num_source_imgs*128*128
            indices = torch.cat((indices_source, indices))
        else:
            indices = indices_source

        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        # pixelbatch is the sample_pixels in one image
        pixelbatch = all_pixels_gt[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()
        
    if store_load and p_idx!=0:
        # store this pseudo images' indices
        torch.save(indices_store, os.path.join(savedir, f"pseudo_{p_idx}.pt"))


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
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
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def generate_pseduo_imgs(args, model, pseudo_poses, hwf, bound):
    pseudo_poses = torch.cat(pseudo_poses)
    pseudo_img = [] #use this to store pseudo_imgs
    ray_origins, ray_directions = get_rays_shapenet(hwf, pseudo_poses)
    for rays_o, rays_d in zip(ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1], args.num_samples, perturb=False)
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape((128,128,3))
            pseudo_img.append(synth.unsqueeze(0))

    # pseudo_img = synth.unsqueeze(0)

    # write img for visual
    # synth = torch.clip(synth, min=0, max=1)
    # synth = (255*synth).to(torch.uint8)
    # synth = synth.cpu().numpy()
    # imageio.imwrite(r'./temp/temp.jpg', synth)
    return pseudo_img


def test():
    # =============== training parameters ==================
    outer_loop = 10
    pseudo_poses_num = 120
    pseudo_poses_range = 2*math.pi
    # ======================================================

    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, default='./configs/shapenet/chairs.json',
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, default='./model_weight/model_1/model_1_meta_epoch15.pth',
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views + args.test_views + args.val_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state = checkpoint['meta_model_state_dict']

    # savedir = Path(args.savedir)
    savedir = Path("./videos_verify_2")
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        print(f"================ Instance {idx+1} ================")

        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        # test imgs are only used for the final test
        # tto_imgs used for training
        tto_imgs, test_imgs, val_imgs = torch.split(imgs, [args.tto_views, args.test_views, args.val_views], dim=0)
        tto_poses, test_poses, val_poses = torch.split(poses, [args.tto_views, args.test_views, args.val_views], dim=0)

        source_imgs = list(torch.split(tto_imgs, 1, dim=0))   # labeled imgs: [img1, img2, ..., (1,128,128,3)]
        source_poses = list(torch.split(tto_poses, 1, dim=0))    # labeled poses: [pose1, pose2, ..., (1, 4, 4)]
        mixed_imgs = copy.deepcopy(source_imgs)     # use these pseudo & labeled images to fine_tune!
        mixed_poses = copy.deepcopy(source_poses)  # use these pseudo & labeled poses to fine_tune!
        inner_steps = [1000]    # use these steps to fine_tune each epoch with different pseudo images

        model.load_state_dict(meta_state)

        p_idx = 0   # p_idx means the number of psuedo images
        # we use pseudo_num epoch to fine_tune the model with pseudo_imgs & labeled images
        for epoch_id in range(1, 1+outer_loop):
            print("Epoch: {}/{}".format(epoch_id, outer_loop))
            # train the outer_model with labeled images
            outer_model = copy.deepcopy(model)
            outer_optim = torch.optim.SGD(outer_model.parameters(), args.tto_lr)

            # Notice: the meaning of epoch_id
            # epoch_id means how many images has been used to train the model
            test_time_optimize(idx, p_idx, args, outer_model, outer_optim, mixed_imgs, mixed_poses, 
                                args.tto_views, hwf, bound, inner_steps[0], True)
            test_psnr = report_result(args, outer_model, test_imgs, test_poses, hwf, bound)
            print("test_psnr: {}".format(test_psnr))
            # the last epoch dont need to generate pseudo_imgs
            if epoch_id != outer_loop:
                # use trained outer_model to generate pseudo_imgs
                pseudo_poses = get_delta_pose_all(source_poses, num_poses=pseudo_poses_num, pose_range=pseudo_poses_range)
                pseudo_imgs = generate_pseduo_imgs(args, outer_model, pseudo_poses, hwf, bound)

                # use these loop to select best pseudo_img's, add them to the train-set
                val_psnrs = []
                for p_img, p_pose in zip(pseudo_imgs, pseudo_poses):
                    # use temp_img/pose to store
                    temp_imgs, temp_poses = copy.deepcopy(mixed_imgs), copy.deepcopy(mixed_poses)
                    temp_imgs.append(p_img)
                    temp_poses.append(p_pose)

                    inner_model = copy.deepcopy(model)
                    inner_optim = torch.optim.SGD(inner_model.parameters(), args.tto_lr)
                    test_time_optimize(idx, p_idx+1, args, inner_model, inner_optim, temp_imgs, temp_poses, 
                                        args.tto_views, hwf, bound, inner_steps[0], False)
                    val_psnr = report_result(args, inner_model, val_imgs, val_poses, hwf, bound)
                    val_psnrs.append(val_psnr.cpu().numpy())
                    print(f"pseudo {p_idx+1}, val_psnr: {val_psnr}")
                
                print("val_psnr_mean: ", np.mean(val_psnrs))

                # sort psnr to choose the best quantity image
                zip_info = zip(val_psnrs, pseudo_imgs, pseudo_poses)
                sorted_info = sorted(zip_info, key=lambda x:x[0], reverse=True)
                val_psnrs, pseudo_imgs, pseudo_poses = zip(*sorted_info)
                mixed_imgs.append(list(pseudo_imgs)[0])
                mixed_poses.append(list(pseudo_poses)[0])

                p_idx += 1

        # copy the last fine_tune model(with the max pseduo imgs) as the final model
        model = copy.deepcopy(outer_model)
        scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)

        create_360_video(args, model, hwf, bound, device, idx+1, savedir, suffix="verify")
        
        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        test_psnrs.append(scene_psnr)

        # write psnr to file
        with open('./videos_verify/psnr.txt','a') as f:
            f.write("video_{}: {}\n".format(idx+1, scene_psnr.cpu().numpy()))
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()