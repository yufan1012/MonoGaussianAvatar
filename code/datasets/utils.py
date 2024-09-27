
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import kornia
from lib.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def modify_theta_phi(xyz, delta_phi=0, delta_theta=0):
    x, y, z = xyz
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, (x**2 + z**2)**0.5) - delta_phi
    theta = np.arctan2(x, z) + delta_theta

    #  Spherical to Cartesian
    x_new = r * np.cos(phi) * np.cos(theta)
    y_new = r * np.sin(phi)
    z_new = r * np.cos(phi) * np.sin(theta)
    return np.array([x_new, y_new, z_new])


def lookat(pos, target, up):
    pos, target, up = np.array(pos), np.array(target), np.array(up)
    fwd = target - pos
    fwd_len = np.sqrt(np.sum(np.power(fwd, 2)))
    fwd = fwd / fwd_len

    right = np.cross(up, fwd)
    right_len = np.sqrt(np.sum(np.power(right, 2)))
    right = right / right_len
    up = np.cross(fwd, right)

    trans = np.array([right, up, fwd])
    pos = -trans.dot(np.array(pos))
    extr = np.zeros((3, 4))
    extr[:3, :3] = trans
    extr[:3, 3] = pos
    return extr


def modify_pitch(extr, look_at_center, down_direction, p=0):
    R = np.linalg.inv(extr[:3, :3]) #w2c
    T = -np.matmul(R, extr[:3, 3])  # camera position

    T = modify_theta_phi(T-look_at_center, delta_phi=p*np.pi/180) + np.array(look_at_center)
    # with open('./cam.obj', 'a') as f:
    #     f.write('v %f %f %f\n' % (T[0], T[1], T[2]))

    lookat_mat = lookat(pos=T, target=look_at_center, up=down_direction)

    return lookat_mat


def get_novel_calib_opencv(data, look_at_center=[0.0, 0.8, 0.0], down_direction=[0, -1, 0], ratio=0.5, p=0, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    intr_list, extr_list = [], []
    data['novel_view'] = {}
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        if p != 0.0:
            extr_new = modify_pitch(extr_new, look_at_center=look_at_center, down_direction=down_direction, p=p)
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)
        intr_list.append(intr_new)
        extr_list.append(extr_new)
    data['novel_view']['intr'] = torch.FloatTensor(np.array(intr_list)).cuda()
    data['novel_view']['extr'] = torch.FloatTensor(np.array(extr_list)).cuda()
    return data


def get_novel_calib_ndc(data, opt, look_at_center=[0.0, 0.8, 0.0], down_direction=[0, -1, 0], ratio=0.5, p=0, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        if p != 0.0:
            extr_new = modify_pitch(extr_new, look_at_center=look_at_center, down_direction=down_direction, p=p)
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, fovX=FovX, fovY=FovY,
                                                K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


# https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/utils.py
def depth2mesh(pts, depth):
    S = depth.shape[3]

    pts = pts.view(S, S, 3).detach().cpu().numpy()
    diff_depth = depth_filter(depth)
    depth[diff_depth > 0.04] = 0
    valid_depth = depth != 0
    valid_depth = kornia.morphology.erosion(valid_depth.float(), torch.ones(3, 3).cuda())
    valid_depth = valid_depth.squeeze().detach().cpu().numpy()

    f_list = []
    for i in range(S-1):
        for j in range(S-1):
            if valid_depth[i, j] and valid_depth[i+1, j+1]:
                if valid_depth[i+1, j]:
                    f_list.append([int(i*S+j+1), int((i+1)*S+j+1), int((i+1)*S+j+2)])
                if valid_depth[i, j+1]:
                    f_list.append([int(i*S+j+1), int((i+1)*S+j+2), int(i*S+j+2)])

    obj_out = trimesh.Trimesh(vertices=pts.reshape(-1, 3), faces=np.array(f_list))

    return obj_out


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def flow2depth(data):
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    offset = torch.broadcast_to(offset[:, None, None, None], data['flow_pred'].shape)
    disparity = offset - data['flow_pred']
    depth = -disparity / data['Tf_x'][:, None, None, None]
    depth *= data['mask'][:, :1, :, :]

    return depth


def depth_filter(depth):
    diff_depth = torch.zeros_like(depth)
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., 1:, :] - depth[..., :-1, :])[..., :-1, 1:-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :-1, :] - depth[..., 1:, :])[..., 1:, 1:-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :, 1:] - depth[..., :, :-1])[..., 1:-1, :-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :, :-1] - depth[..., :, 1:])[..., 1:-1, 1:]

    return diff_depth


def perspective(pts, calibs):
    # pts: [B, N, 3]
    # calibs: [B, 3, 4]
    pts = pts.permute(0, 2, 1)
    pts = torch.bmm(calibs[:, :3, :3], pts)
    pts = pts + calibs[:, :3, 3:4]
    pts[:, :2, :] /= pts[:, 2:, :]
    pts = pts.permute(0, 2, 1)
    return pts


def depth_projection(pts, valid, target_intr, target_extr, S=1024):
    '''
    input:
        pts: [B, S*S, 3]
        valid: [B, S*S]
    output:
        depth_out: [B, 1, S, S]
    '''
    target_calib = torch.matmul(target_intr, target_extr)
    pts_valid = torch.zeros_like(pts)
    pts_valid[valid] = pts[valid]

    pts_valid = perspective(pts_valid, target_calib)
    pts_valid[:, :, 2:] = 1.0 / (pts_valid[:, :, 2:] + 1e-8)
    pts_valid[:, :, :2] = torch.clamp(pts_valid[:, :, :2], 0, S-1)

    B = pts.shape[0]
    depth_out = torch.zeros([B, 1, S, S], device=pts.device)
    depth_out[:, 0, pts_valid[:, :, 1].int(), pts_valid[:, :, 0].int()] = pts_valid[:, :, 2]

    return depth_out.detach()
