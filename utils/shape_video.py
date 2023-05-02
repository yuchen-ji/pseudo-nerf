import math
from turtle import pos
import imageio
import torch
from models.rendering import get_rays_shapenet, sample_points, volume_render


def create_posemat(radius, theta, phi):
    """
    3d transformations to create pose matrix from radius, theta and phi
    """
    trans_t = lambda t : torch.as_tensor([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, t],
                            [0, 0, 0, 1],
                            ], dtype=torch.float)

    rot_theta = lambda theta : torch.as_tensor([
                                    [torch.cos(theta), 0, -torch.sin(theta), 0],
                                    [0, 1, 0, 0],
                                    [torch.sin(theta), 0, torch.cos(theta), 0],
                                    [0, 0, 0, 1],
                                    ], dtype=torch.float)

    rot_phi = lambda phi : torch.as_tensor([
                                [1, 0, 0, 0],
                                [0, torch.cos(phi), -torch.sin(phi), 0],
                                [0, torch.sin(phi), torch.cos(phi), 0],
                                [0, 0, 0, 1],
                                ], dtype=torch.float)
        
    pose = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    pose = torch.as_tensor([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]
                            ], dtype=torch.float) @ pose
    return pose


def get_delta_pose(pose, device, theta=0, phi=0):
    convert = torch.as_tensor([[-1, 0, 0, 0],
                                [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]
                               ], dtype=torch.float).to(device)

    rot_theta = lambda theta : torch.as_tensor([
                                    [torch.cos(theta), 0, -torch.sin(theta), 0],
                                    [0, 1, 0, 0],
                                    [torch.sin(theta), 0, torch.cos(theta), 0],
                                    [0, 0, 0, 1],
                                    ], dtype=torch.float)

    rot_phi = lambda phi : torch.as_tensor([
                                [1, 0, 0, 0],
                                [0, torch.cos(phi), -torch.sin(phi), 0],
                                [0, torch.sin(phi), torch.cos(phi), 0],
                                [0, 0, 0, 1],
                                ], dtype=torch.float)

    if theta.dtype != torch.float:
        theta = torch.tensor(theta).float()
        phi = torch.tensor(phi).float()
    
    if isinstance(pose, list):
        pose = torch.cat(pose, dim=0)

    pose = pose.to(device)
    # here poses are not 1 pose, use broadcast to rotate the poses
    init_pose = convert @ pose
    pose = rot_theta(theta).to(device) @ rot_phi(phi).to(device) @ init_pose
    pose = convert @ pose
    pose = list(torch.split(pose, 1, dim=0))
    return pose


def get_delta_pose_all(pose, device, num_poses, pose_range=2*math.pi):
    all_poses = []
    for delta in torch.linspace(0, pose_range, num_poses+1)[1:]:
        # theta = phi = delta
        all_poses.extend(get_delta_pose(pose, device, delta, delta))
    return all_poses


def get_360_poses(radius=4, phi=math.pi/5, num_poses=120):
    """
    create spherical camera poses for 360 view around the scene
    """
    radius = torch.as_tensor(radius, dtype=torch.float)
    phi = torch.as_tensor(-phi, dtype=torch.float)

    all_poses = []
    for theta in torch.linspace(0, 2*math.pi, num_poses+1)[:-1]:
        all_poses.append(create_posemat(radius, theta, phi))
    all_poses = torch.stack(all_poses, dim=0)
    return all_poses


def create_360_video(args, model, hwf, bound, device, scene_id, savedir, suffix='init'):
    """
    create 360 video of a specific shape
    """
    video_frames = []
    poses_360 = get_360_poses(args.radius).to(device)
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses_360)

    for rays_o, rays_d in zip(ray_origins, ray_directions):
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
            synth = torch.cat(synth, dim=0).reshape(int(hwf[0]), int(hwf[1]), 3)
            synth = torch.clip(synth, min=0, max=1)
            synth = (255*synth).to(torch.uint8)
            video_frames.append(synth)
    video_frames = torch.stack(video_frames, dim=0)
    video_frames = video_frames.cpu().numpy()

    video_path = savedir.joinpath(f"{scene_id}_{suffix}.mp4")
    imageio.mimwrite(video_path, video_frames, fps=30)

    return None