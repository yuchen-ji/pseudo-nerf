from cmath import e
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4 5 6 7'
sys.path.insert(0, '/home/shenxi/interns/JiYuchen/semi-nerf/')

from email.mime import image
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video, get_delta_pose, get_delta_pose_all
from models.rendering import get_rays_shapenet, sample_points, volume_render
import copy
import math
import imageio


def test_time_optimize(args, model, optim, imgs, poses, hwf, bound, inner_steps, epoch):
    """
    test-time-optimize the meta trained model on available views
    """
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
        indices = torch.randint(num_rays, size=[num_imgs * args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        # pixelbatch is the sample_pixels in one image
        pixelbatch = all_pixels_gt[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)

        # her we store the loss weight v & indice_split for each image
        weight = []
        indice_split = [torch.where((indices>=i*128*128) & (indices<(i+1)*128*128)) for i in range(num_imgs)]
        if epoch > 1:
            # here we first only use pseudo to train, then use the labeled in last steps
            for i in range(num_imgs):
                num_pixel = len(indice_split[i][0])
                if step <= 700:
                    weight.extend([[0, 0, 0]] * num_pixel) if i < 4 else weight.extend([[1, 1, 1]] * num_pixel)
                else:
                    weight.extend([[1, 1, 1]] * num_pixel) if i < 4 else weight.extend([[0, 0, 0]] * num_pixel) 

        weight = torch.as_tensor(weight, dtype=torch.float).to(args.device)          

        loss = F.mse_loss(colors, pixelbatch) if epoch ==1 else F.mse_loss(weight*colors, weight*pixelbatch)
        loss.backward()
        optim.step()


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
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, default='./configs/shapenet/chairs.json',
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, default='./weights/meta_epoch15.pth',
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.tto_views = 4
    args.test_views = 25
    args.device = device

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state = checkpoint['meta_model_state_dict']

    # savedir = Path(args.savedir)
    savedir = Path("./videos_4v_4v_pseudo_first")
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        # test imgs are only used for the final test
        # tto_imgs used for training
        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        source_imgs = list(torch.split(tto_imgs, 1, dim=0))   # labeled imgs: [img1, img2, ..., (1,128,128,3)]
        source_poses = list(torch.split(tto_poses, 1, dim=0))    # labeled poses: [pose1, pose2, ..., (1, 4, 4)]
        mixed_imgs = copy.deepcopy(source_imgs)     # use these pseudo & labeled images to fine_tune!
        mixed_poses = copy.deepcopy(source_poses)  # use these pseudo & labeled poses to fine_tune!
        inner_steps = [1000]    # use these steps to fine_tune each epoch with different pseudo images

        model.load_state_dict(meta_state)
        # optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        pseudo_num = 4
        # we use pseudo_num epoch to fine_tune the model with pseudo_imgs & labeled images
        for epoch_id in range(pseudo_num + 1):
            print("Epoch: {}/{} epoch".format(epoch_id+1, pseudo_num+1))
            temp_model = copy.deepcopy(model)
            optim = torch.optim.SGD(temp_model.parameters(), args.tto_lr)
            test_time_optimize(args, temp_model, optim, mixed_imgs, mixed_poses, hwf, bound, inner_steps[0], epoch=epoch_id+1)

            # if not the last epoch, then generate pseudo_imgs
            if epoch_id+1 != pseudo_num+1:
                pseudo_pose = get_delta_pose_all(source_poses, device, num_poses=(epoch_id+1), pose_range=((epoch_id+1)*math.pi/4))
                pseudo_img = generate_pseduo_imgs(args, temp_model, pseudo_pose, hwf, bound)
                mixed_imgs = copy.deepcopy(source_imgs)
                mixed_poses = copy.deepcopy(source_poses)
                mixed_imgs.extend(pseudo_img)
                mixed_poses.extend(pseudo_pose)

        # copy the last fine_tune model(with the max pseduo imgs) as the final model
        model = copy.deepcopy(temp_model)
        scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)

        create_360_video(args, model, hwf, bound, device, idx+1, savedir)
        
        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        test_psnrs.append(scene_psnr)

        # write psnr to file
        with open('./videos_4v_4v_pseudo_first/psnr.txt','a') as f:
            f.write("video_{}: {}\n".format(idx+1, scene_psnr.cpu().numpy()))
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()