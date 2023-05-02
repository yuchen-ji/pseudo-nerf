import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
import torch.nn as nn
import cv2 as cv


def test_time_optimize(args, model, optim, imgs, poses, hwf, bound, idx):
    """
    test-time-optimize the meta trained model on available views
    """
    pixels = imgs.reshape(-1, 3)

    generate_samplimg_pixels = True    # means load sampling_pixels True

    # get groud truth image's rays_o & rays_d
    # i think rays_d is defined as the rays direction, along(x-direction, y-direction, z-direction)
    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    # num_rays is 128*128, which is the number of rays
    num_rays = rays_d.shape[0]

    if generate_samplimg_pixels:
        # store each instance's sampling pixels
        indices_range = []
    else:
        indices_range = torch.load('./sampling_pixels/1v_128p/instance_{}.pt'.format(idx+1))

    for step in range(args.tto_steps):
        # 128*128 rays are too many, so that we should sample a little rays for calculate loss
        if generate_samplimg_pixels:
            indices = torch.randint(num_rays, size=[args.tto_batchsize * 8])
        else:
            indices = indices_range[step]
            
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        # pixelbatch is the sample_pixels in one image
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        if generate_samplimg_pixels:
            indices_range.append(indices)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()
    
    # store the indices
    if generate_samplimg_pixels:
        torch.save(indices_range, './sampling_pixels/1v_512p/instance_{}.pt'.format(idx+1))

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


def test():
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
                            num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    # use multiple GPU to train the model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state = checkpoint['meta_model_state_dict']

    savedir = Path("./videos_1v_wo")
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
        
        # for i in range(imgs.shape[0]):
        #     mae_img = imgs[i].cpu().numpy()
        #     cv.imwrite(f"mae_test/shapenet_{i}.jpg", mae_img)

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
        
        # load model weight which is single GPU saved
        if isinstance(model, torch.nn.DataParallel):
            model.state_dict = meta_state
        else:
            model.load_state_dict(meta_state)

        optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        test_time_optimize(args, model, optim, tto_imgs, tto_poses, hwf, bound, idx)
        scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)

        # create_360_video(args, model, hwf, bound, device, idx+1, savedir)
        
        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        test_psnrs.append(scene_psnr)

        # write psnr to file
        with open('./videos_1v_wo/psnr.txt','a') as f:
            f.write("video_{}: {}\n".format(idx+1, scene_psnr.cpu().numpy()))
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()