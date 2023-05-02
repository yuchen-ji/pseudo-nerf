import torch
import math


def get_delta_pose_all(pose, num_poses, pose_range=2*math.pi):
    all_poses = []
    delta_split = torch.linspace(0, pose_range, num_poses+1)[1:] if pose_range != 2*math.pi else torch.linspace(0, pose_range, num_poses+1)[1:-1]
    for delta in delta_split:
        # theta = phi = delta
        all_poses.extend(get_delta_pose(pose, delta, delta))
        # x axis: phi
        # all_poses.extend(get_delta_pose(pose, theta=torch.tensor(0), phi=delta))
        # z axis: theta
        # all_poses.extend(get_delta_pose(pose, theta=delta, phi=torch.tensor(0)))

    return all_poses


def get_delta_pose(pose, theta=0, phi=0):
    # covert used to trans the pose to pose_init, which in another coordinate system(can use rot_theta & rot_phi to trans pose_init)
    convert = torch.as_tensor([[-1, 0, 0, 0],
                                [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]
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

    if theta.dtype != torch.float:
        theta = torch.tensor(theta).float()
        phi = torch.tensor(phi).float()
    
    if isinstance(pose, list):
        pose = torch.cat(pose, dim=0)

    # caculate the rotate angle
    rot_t = rot_theta(theta)
    rot_p = rot_phi(phi)

    if pose.is_cuda:
        convert = convert.to(pose.device)
        rot_t = rot_t.to(pose.device)
        rot_p = rot_p.to(pose.device)
        
    # here poses are not 1 pose, use broadcast to rotate the poses
    init_pose = convert @ pose
    pose = rot_t @ rot_p @ init_pose
    pose = convert @ pose
    pose = list(torch.split(pose, 1, dim=0))
    return pose