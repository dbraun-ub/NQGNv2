# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset, MonoDatasetEvaluation

import oxts_utils


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, ".txt")
        oxts_path = os.path.join(self.data_path, folder, "oxts/data", f_str)
        oxts = oxts_utils.load_oxts_packets_and_poses([oxts_path])
        if oxts is []:
            P = np.eye(4)
        else:
            P = oxts[0][1]
        # From oxts orientation to camera orientation with z axes facing forward

        # IMU to velodyne
        R_imu2vel = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
                      [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                      [2.024406e-03, 1.482454e-02, 9.998881e-01]])
        t_imu2vel = np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
        P_imu2vel = np.eye(4)
        P_imu2vel[:3,:3] = R_imu2vel
        P_imu2vel[:3,3] = t_imu2vel

        # Velodyne to camera 0
        R_velo2cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                       [1.480249e-02, 7.280733e-04, -9.998902e-01],
                       [9.998621e-01, 7.523790e-03, 1.480755e-02]])
        t_velo2cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
        P_velo2cam = np.eye(4)
        P_velo2cam[:3,:3] = R_velo2cam
        P_velo2cam[:3,3] = t_velo2cam

        # Camera 0 to camera 2
        R_02 = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                 [5.251945e-03, 9.999804e-01, -3.413835e-03],
                 [4.570332e-03, 3.389843e-03, 9.999838e-01]])
        t_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])
        P_02 = np.eye(4)
        P_02[:3,:3] = R_02
        P_02[:3,3] = t_02
        P_imu2cam2 = np.matmul(P_02, np.matmul(P_velo2cam, P_imu2vel))

        P = np.matmul(P, np.linalg.inv(P_imu2cam2))

        return P

class KITTIDatasetEvaluation(MonoDatasetEvaluation):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDatasetEvaluation, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDatasetEvaluation(KITTIDatasetEvaluation):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDatasetEvaluation, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, ".txt")
        oxts_path = os.path.join(self.data_path, folder, "oxts/data", f_str)
        oxts = oxts_utils.load_oxts_packets_and_poses([oxts_path])
        if oxts is []:
            P = np.eye(4)
        else:
            P = oxts[0][1]
        # From oxts orientation to camera orientation with z axes facing forward

        # IMU to velodyne
        R_imu2vel = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
                      [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                      [2.024406e-03, 1.482454e-02, 9.998881e-01]])
        t_imu2vel = np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
        P_imu2vel = np.eye(4)
        P_imu2vel[:3,:3] = R_imu2vel
        P_imu2vel[:3,3] = t_imu2vel

        # Velodyne to camera 0
        R_velo2cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                       [1.480249e-02, 7.280733e-04, -9.998902e-01],
                       [9.998621e-01, 7.523790e-03, 1.480755e-02]])
        t_velo2cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
        P_velo2cam = np.eye(4)
        P_velo2cam[:3,:3] = R_velo2cam
        P_velo2cam[:3,3] = t_velo2cam

        # Camera 0 to camera 2
        R_02 = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                 [5.251945e-03, 9.999804e-01, -3.413835e-03],
                 [4.570332e-03, 3.389843e-03, 9.999838e-01]])
        t_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])
        P_02 = np.eye(4)
        P_02[:3,:3] = R_02
        P_02[:3,3] = t_02
        P_imu2cam2 = np.matmul(P_02, np.matmul(P_velo2cam, P_imu2vel))

        P = np.matmul(P, np.linalg.inv(P_imu2cam2))

        return P

class KITTI2012Dataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTI2012Dataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        # f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # print(f"self.data_path: {self.data_path}")
        # print(f"folder: {folder}")

        image_path = os.path.join(self.data_path, "image_2", folder)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        disp_path = os.path.join("/media/HDD1/train_daniel/kitti2012_disparity/testing", folder)

        disp_gt = cv2.imread(disp_path, 0)
        # We use the same baseline used to train monodepth
        baseline = 0.1
        focale = self.K[0,0] * disp_gt.shape[1]

        depth_gt = baseline * focale / disp_gt



        # depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def check_depth(self):
        return True


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
