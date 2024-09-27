import os
import torch
import numpy as np
import cv2
import json
import imageio
import skimage
from tqdm import tqdm

from model.cameras import MiniCam, getProjectionMatrix, getWorld2View2, focal2fov



class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 is_eval,
                 subsample=1,
                 hard_mask=False,
                 only_json=False,
                 use_mean_expression=False,
                 use_var_expression=False,
                 use_background=False,
                 load_images=False,
                 ):

        sub_dir = [str(dir) for dir in sub_dir]
        self.img_res = img_res
        self.use_background = use_background
        self.load_images = load_images
        self.hard_mask = hard_mask

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            "world_mats": [],
            "expressions": [],
            "flame_pose": [],
            "img_name": [],
            "sub_dir": [],
            "world_view_transform": [],
            "full_proj_transform": [],
            "camera_center": [],
            "tanfovx": [],
            "tanfovy": [],
            "bg_color": [],
        }

        for dir in sub_dir:
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory {} is empty".format(instance_dir)

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)
            # gaussian intr
            intrinc_ga = camera_dict['intrinsics']
            intrinsics_ga = np.zeros((3, 3))
            intrinsics_ga[0, 0] = -intrinc_ga[0] * self.img_res[0]
            intrinsics_ga[1, 1] = intrinc_ga[1] * self.img_res[1]
            intrinsics_ga[2, 2] = 1
            intrinsics_ga[0, 2] = intrinc_ga[2] * self.img_res[0]
            intrinsics_ga[1, 2] = intrinc_ga[3] * self.img_res[1]
            self.intrinsics_ga = intrinsics_ga
            for frame in camera_dict['frames']:
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(world_mat)
                camera_pose = np.vstack((np.array(frame['world_mat']), np.array([[0, 0, 0, 1]])))
                c2w = np.linalg.inv(camera_pose)
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                trans = np.array([0.0, 0.0, 0.0], np.float32)
                scale = 1.0
                world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(trans), scale)).transpose(0, 1)
                bg_color = np.array([1., 1., 1.], np.float32)
                bg_color = torch.from_numpy(bg_color)
                self.bg_color = bg_color
                fx = self.intrinsics_ga[0, 0]
                fy = self.intrinsics_ga[1, 1]
                # bg_mask = torch.ones(3, 512, 512)
                FovX = focal2fov(fx, self.img_res[1])
                FovX = np.array(FovX, np.float32)
                FovX = torch.from_numpy(FovX)
                tanfovx = torch.tan(FovX * 0.5).item()
                FovY = focal2fov(fy, self.img_res[0])
                FovY = np.array(FovY, np.float32)
                FovY = torch.from_numpy(FovY)
                tanfovy = torch.tan(FovY * 0.5).item()
                znear = 0.01
                zfar = 100

                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY,
                                                        K=self.intrinsics_ga,
                                                        h=self.img_res[1], w=self.img_res[0]).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                world_view_transform = world_view_transform
                self.data["tanfovx"].append(tanfovx)
                self.data["tanfovy"].append(tanfovy)
                self.data["bg_color"].append(bg_color)
                # self.data["bg_mask"].append(bg_mask)
                self.data["world_view_transform"].append(world_view_transform)
                self.data["full_proj_transform"].append(full_proj_transform)
                self.data["camera_center"].append(camera_center)
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                self.data["sub_dir"].append(dir)
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))


        self.gt_dir = instance_dir
        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
        elif isinstance(subsample, list):
            if len(subsample) == 2:
                subsample = list(range(subsample[0], subsample[1]))
            for k, v in self.data.items():
                self.data[k] = [v[s] for s in subsample]

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()

        intrinsics = np.zeros((4, 4))
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics
        # print(intrinsics)
        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1


        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

        images = []
        masks = []
        if load_images and not only_json:
            print("Loading all images, this might take a while.")
            for idx in tqdm(range(len(self.data["image_paths"]))):
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1,0)).float()
                object_mask = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                if not self.use_background:
                    if not hard_mask:
                        rgb = rgb * object_mask.unsqueeze(1).float() + (1 - object_mask.unsqueeze(1).float())
                        # mask = (object_mask != 0).float()
                    else:
                        rgb = rgb * (object_mask.unsqueeze(1) > 0.5) + ~(object_mask.unsqueeze(1) > 0.5)
                        # mask = (object_mask != 0).float()
                images.append(rgb)
                masks.append(object_mask)
                # masks.append(mask)

        self.data['images'] = images
        self.data['masks'] = masks
        # self.data['masks'] = masks



    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx],
            "world_view_transform": self.data["world_view_transform"][idx],
            "full_proj_transform": self.data["full_proj_transform"][idx],
            "camera_center": self.data["camera_center"][idx],
            "tanfovx": self.data["tanfovx"][idx],
            "tanfovy": self.data["tanfovy"][idx],
            "bg_color": self.data["bg_color"][idx],
            # 'bg_mask': self.data["bg_mask"][idx],
            }

        ground_truth = {}

        if not self.only_json:
            if not self.load_images:
                ground_truth["object_mask"] = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                if not self.use_background:
                    if not self.hard_mask:
                        ground_truth['rgb'] = rgb * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                    else:
                        ground_truth['rgb'] = rgb * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                else:
                    ground_truth['rgb'] = rgb
            else:
                ground_truth = {
                    'rgb': self.data['images'][idx],
                    'object_mask': self.data['masks'][idx]
                }

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

def load_rgb(path, img_res):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path, img_res):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)

    alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha / 255

    return object_mask