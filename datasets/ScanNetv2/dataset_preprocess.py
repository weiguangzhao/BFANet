# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/29  下午1:50
# File Name: dataset_preprocess.py
# IDE: PyCharm

import math
import ocnn
import torch
import random
import numpy as np
import SharedArray as SA
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
from ocnn.octree import Points, Octree
import MinkowskiEngine as ME
import datasets.ScanNetv2.transform as aug_transform



class Dataset:
    def __init__(self, cfg):

        self.batch_size = cfg.batch_size
        self.batch_size_v = cfg.batch_size_v
        self.dataset_workers = cfg.num_works
        self.cache = cfg.cache
        self.dist = cfg.dist
        self.voxel_size = cfg.voxel_size
        self.mixup = cfg.mixup
        self.backbone = cfg.backbone

        # ####Data augment for coord and normals
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomDropout = aug_transform.RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2)
        self.crop_p = cfg.crop_p
        self.SphereCrop = aug_transform.SphereCrop(point_max=self.crop_p, mode='random')
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5)
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1 / 64, 1 / 64], axis="y", p=0.5)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1 / 64, 1 / 64], axis="x", p=0.5)
        self.RandomScale = aug_transform.RandomScale(scale=[0.9, 1.1])
        self.RandomFlip = aug_transform.RandomFlip(p=0.5)
        self.RandomJitter = aug_transform.RandomJitter(sigma=0.005, clip=0.02)
        self.ElasticDistortion = aug_transform.ElasticDistortion(distortion_params=[[0.2, 0.4], [0.8, 1.6]])
        self.MixupScene = aug_transform.MixupScene()

        # ####Data augment for color
        self.ChromaticAutoContrast = aug_transform.ChromaticAutoContrast(p=0.2, blend_factor=None)
        self.ChromaticTranslation = aug_transform.ChromaticTranslation(p=0.95, ratio=0.05)
        self.ChromaticJitter = aug_transform.ChromaticJitter(p=0.95, std=0.05)
        self.NormalizeColor = aug_transform.NormalizeColor()

        self.train_aug_compose = aug_transform.Compose([self.CenterShift, self.RandomDropout, self.SphereCrop,
                                                        self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                        self.RandomScale, self.RandomFlip, self.ElasticDistortion,
                                                        self.ChromaticAutoContrast, self.ChromaticTranslation,
                                                        self.ChromaticJitter, self.NormalizeColor, self.CenterShift])
        self.val_aug_compose = aug_transform.Compose([self.CenterShift, self.NormalizeColor, self.RandomRotate_z, self.CenterShift])

        self.dataset_root = 'datasets'
        self.dataset = 'ScanNetv2'
        self.dataset_suffix = '.npy'
        self.npy_dir = 'datasets/ScanNetv2/npy/'

        # ####seprate the train val test set
        self.train_file_list = np.loadtxt('datasets/ScanNetv2/scannetv2_train.txt', dtype=str)
        self.val_file_list = np.loadtxt('datasets/ScanNetv2/scannetv2_val.txt', dtype=str)
        self.test_file_list = np.loadtxt('datasets/ScanNetv2/scannetv2_test.txt', dtype=str)
        self.train_file_list.sort()
        self.val_file_list.sort()
        self.test_file_list.sort()

        # ####FOR LONG TRAINING
        # self.train_file_list = np.repeat(self.train_file_list, 4, axis=0)

    def trainLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
                                            drop_last=True, pin_memory=False,
                                            worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        val_set = list(range(len(self.val_file_list)))
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size_v, collate_fn=self.valMerge,
                                          num_workers=self.dataset_workers,
                                          shuffle=False, sampler=None, drop_last=False, pin_memory=True,
                                          worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        test_set = list(range(len(self.test_file_list)))
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_set) if self.dist else None
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size_v, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None, drop_last=False, pin_memory=True,
                                           worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def trainMerge(self, id):

        points_list = []
        inbox_mask_list = []
        file_name = []
        xyz_original = []
        sem_label = []
        batch_offset = []

        xyz_voxel = []
        feat_voxel = []
        v2p_index_batch = []
        total_voxel_num = 0
        for i, idx in enumerate(id):
            fn = self.train_file_list[idx]  # get shm name
            if self.cache:
                coord = SA.attach("shm://v2_{}_coord".format(fn)).copy()
                color = SA.attach("shm://v2_{}_color".format(fn)).copy()
                normal = SA.attach("shm://v2_{}_normal".format(fn)).copy()
                segment = SA.attach("shm://v2_{}_segment".format(fn)).copy()
            else:
                coord = np.load(self.npy_dir + 'train/{}/coord.npy'.format(fn))
                color = np.load(self.npy_dir + 'train/{}/color.npy'.format(fn))
                normal = np.load(self.npy_dir + 'train/{}/normal.npy'.format(fn))
                segment = np.load(self.npy_dir + 'train/{}/segment.npy'.format(fn))[:, 0]
                pass

            file_name.append(self.train_file_list[idx])
            Point_dict_1 = {'coord': coord, 'color': color, 'normal': normal, 'segment': segment}
            Point_dict_1 = self.train_aug_compose(Point_dict_1)

            # #####mix up
            if self.mixup == True and np.random.rand() < 0.90:
                mix_id = np.floor(np.random.rand() * len(self.train_file_list)).astype(np.int64)
                mix_fn = self.train_file_list[mix_id]
                mix_coord = SA.attach("shm://v2_{}_coord".format(mix_fn)).copy()
                mix_color = SA.attach("shm://v2_{}_color".format(mix_fn)).copy()
                mix_normal = SA.attach("shm://v2_{}_normal".format(mix_fn)).copy()
                mix_segment = SA.attach("shm://v2_{}_segment".format(mix_fn)).copy()

                Point_dict_2 = {'coord': mix_coord, 'color': mix_color, 'normal': mix_normal, 'segment': mix_segment}
                Point_dict_2 = self.train_aug_compose(Point_dict_2)
                Point_dict_1 = self.MixupScene(Point_dict_1, Point_dict_2)
                Point_dict_1 = self.SphereCrop(Point_dict_1)
                Point_dict_1 = self.CenterShift(Point_dict_1)

            if self.backbone == "octformer":
                # ######scale for Octree
                Point_dict_1['coord'] = Point_dict_1['coord']/10.25
                points = Points(torch.from_numpy(Point_dict_1['coord']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['normal']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['color']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['segment']).type(torch.int64))
                inbox_mask = points.clip(min=-1, max=1)
                # assert False not in inbox_mask

            elif self.backbone == "mink":
                # ------------------------------- Voxel and Batch -------------------------
                feats_rgb_normal_line = np.concatenate((Point_dict_1['color'], Point_dict_1['normal']), axis=1).astype(np.float32)
                quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(Point_dict_1['coord'], feats_rgb_normal_line, \
                                                                                             quantization_size=self.voxel_size,
                                                                                             return_index=True,
                                                                                             return_inverse=True)
                v2p_index = inverse_index + total_voxel_num
                total_voxel_num = total_voxel_num + index.shape[0]

            # -------------------------------Batch -------------------------
            xyz_original.append(torch.from_numpy(Point_dict_1['coord']).type(torch.float32))
            sem_label.append(torch.from_numpy(Point_dict_1['segment']).type(torch.int64))
            batch_offset.append(Point_dict_1['coord'].shape[0])

            if self.backbone == "octformer":
                points_list.append(points)
                inbox_mask_list.append(inbox_mask)
            elif self.backbone == "mink":
                xyz_voxel.append(quantized_coords)
                feat_voxel.append(feats_all)
                v2p_index_batch.append(v2p_index)

        # ####numpy to torch
        batch_offset = torch.from_numpy(np.array(batch_offset))
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        sem_label = torch.cat(sem_label, 0).to(torch.int64)

        # #### retrun
        if self.backbone == "octformer":
            inbox_mask = torch.cat(inbox_mask_list, 0)
            return {'fn': file_name, 'xyz_original': xyz_original, 'sem_label': sem_label, 'batch_offset': batch_offset,
                    'points': points_list, 'inbox_mask': inbox_mask}

        elif self.backbone == "mink":
            xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
            v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
            return {'fn': file_name, 'xyz_original': xyz_original, 'sem_label': sem_label, 'batch_offset': batch_offset,
                    'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'v2p_index': v2p_index_batch}

    def valMerge(self, id):

        points_list = []
        inbox_mask_list = []
        file_name = []
        xyz_original = []
        sem_label = []
        batch_offset = []

        xyz_voxel = []
        feat_voxel = []
        v2p_index_batch = []
        total_voxel_num = 0
        id = id + id + id
        # id = id
        for i, idx in enumerate(id):
            fn = self.val_file_list[idx]  # get shm name
            if self.cache:
                coord = SA.attach("shm://v2_{}_coord".format(fn)).copy()
                color = SA.attach("shm://v2_{}_color".format(fn)).copy()
                normal = SA.attach("shm://v2_{}_normal".format(fn)).copy()
                segment = SA.attach("shm://v2_{}_segment".format(fn)).copy()
                superpoint = SA.attach("shm://v2_{}_superpoint".format(fn)).copy()
            else:
                coord = np.load(self.npy_dir + 'train/{}/coord.npy'.format(fn))
                color = np.load(self.npy_dir + 'train/{}/color.npy'.format(fn))
                normal = np.load(self.npy_dir + 'train/{}/normal.npy'.format(fn))
                segment = np.load(self.npy_dir + 'train/{}/segment.npy'.format(fn))[:, 0]
                superpoint = np.load(self.npy_dir + 'train/{}/superpoint.npy'.format(fn))
                pass

            file_name.append(self.val_file_list[idx])
            Point_dict_1 = {'coord': coord, 'color': color, 'normal': normal, 'segment': segment}
            Point_dict_1 = self.val_aug_compose(Point_dict_1)

            if self.backbone == "octformer":
                # ######scale for Octree
                Point_dict_1['coord'] = Point_dict_1['coord'] / 10.25
                points = Points(torch.from_numpy(Point_dict_1['coord']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['normal']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['color']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['segment']).type(torch.int64))
                inbox_mask = points.clip(min=-1, max=1)
                # assert False not in inbox_mask

            elif self.backbone == "mink":
                # ------------------------------- Voxel and Batch -------------------------
                feats_rgb_normal_line = np.concatenate((Point_dict_1['color'], Point_dict_1['normal']), axis=1).astype(
                    np.float32)
                quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(Point_dict_1['coord'],
                                                                                             feats_rgb_normal_line, \
                                                                                             quantization_size=self.voxel_size,
                                                                                             return_index=True,
                                                                                             return_inverse=True)
                v2p_index = inverse_index + total_voxel_num
                total_voxel_num = total_voxel_num + index.shape[0]

            # -------------------------------Batch -------------------------
            xyz_original.append(torch.from_numpy(Point_dict_1['coord']).type(torch.float32))
            sem_label.append(torch.from_numpy(Point_dict_1['segment']).type(torch.int64))
            batch_offset.append(Point_dict_1['coord'].shape[0])

            if self.backbone == "octformer":
                points_list.append(points)
                inbox_mask_list.append(inbox_mask)
            elif self.backbone == "mink":
                xyz_voxel.append(quantized_coords)
                feat_voxel.append(feats_all)
                v2p_index_batch.append(v2p_index)

        # ####numpy to torch
        superpoint = torch.from_numpy(superpoint)
        batch_offset = torch.from_numpy(np.array(batch_offset))
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        sem_label = torch.cat(sem_label, 0).to(torch.int64)

        # #### retrun
        if self.backbone == "octformer":
            inbox_mask = torch.cat(inbox_mask_list, 0)
            return {'fn': file_name, 'xyz_original': xyz_original, 'sem_label': sem_label, 'batch_offset': batch_offset,
                    'sup': superpoint, 'points': points_list, 'inbox_mask': inbox_mask}

        elif self.backbone == "mink":
            xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
            v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
            return {'fn': file_name, 'xyz_original': xyz_original, 'sem_label': sem_label, 'batch_offset': batch_offset,
                    'sup': superpoint, 'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch,
                    'v2p_index': v2p_index_batch}

    def testMerge(self, id):

        points_list = []
        inbox_mask_list = []
        file_name = []
        xyz_original = []
        batch_offset = []

        xyz_voxel = []
        feat_voxel = []
        v2p_index_batch = []
        total_voxel_num = 0
        id = id + id + id
        for i, idx in enumerate(id):
            fn = self.test_file_list[idx]  # get shm name
            if self.cache:
                coord = SA.attach("shm://v2_{}_coord".format(fn)).copy()
                color = SA.attach("shm://v2_{}_color".format(fn)).copy()
                normal = SA.attach("shm://v2_{}_normal".format(fn)).copy()
                superpoint = SA.attach("shm://v2_{}_superpoint".format(fn)).copy()
            else:
                coord = np.load(self.npy_dir + 'train/{}/coord.npy'.format(fn))
                color = np.load(self.npy_dir + 'train/{}/color.npy'.format(fn))
                normal = np.load(self.npy_dir + 'train/{}/normal.npy'.format(fn))
                superpoint = np.load(self.npy_dir + 'train/{}/superpoint.npy'.format(fn))
                pass

            file_name.append(self.test_file_list[idx])
            Point_dict_1 = {'coord': coord, 'color': color, 'normal': normal}
            Point_dict_1 = self.val_aug_compose(Point_dict_1)

            if self.backbone == "octformer":
                # ######scale for Octree
                Point_dict_1['coord'] = Point_dict_1['coord'] / 10.25
                points = Points(torch.from_numpy(Point_dict_1['coord']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['normal']).type(torch.float32),
                                torch.from_numpy(Point_dict_1['color']).type(torch.float32))
                inbox_mask = points.clip(min=-1, max=1)
                assert False not in inbox_mask

            elif self.backbone == "mink":
                # ------------------------------- Voxel and Batch -------------------------
                feats_rgb_normal_line = np.concatenate((Point_dict_1['color'], Point_dict_1['normal']),
                                                       axis=1).astype(
                    np.float32)
                quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(Point_dict_1['coord'],
                                                                                             feats_rgb_normal_line, \
                                                                                             quantization_size=self.voxel_size,
                                                                                             return_index=True,
                                                                                             return_inverse=True)
                v2p_index = inverse_index + total_voxel_num
                total_voxel_num = total_voxel_num + index.shape[0]

            # -------------------------------Batch -------------------------
            xyz_original.append(torch.from_numpy(Point_dict_1['coord']).type(torch.float32))
            batch_offset.append(Point_dict_1['coord'].shape[0])

            if self.backbone == "octformer":
                points_list.append(points)
                inbox_mask_list.append(inbox_mask)
            elif self.backbone == "mink":
                xyz_voxel.append(quantized_coords)
                feat_voxel.append(feats_all)
                v2p_index_batch.append(v2p_index)

        # ####numpy to torch
        superpoint = torch.from_numpy(superpoint)
        batch_offset = torch.from_numpy(np.array(batch_offset))
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)

        # #### retrun
        if self.backbone == "octformer":
            inbox_mask = torch.cat(inbox_mask_list, 0)
            return {'fn': file_name, 'xyz_original': xyz_original,
                    'batch_offset': batch_offset,
                    'sup': superpoint, 'points': points_list, 'inbox_mask': inbox_mask}

        elif self.backbone == "mink":
            xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
            v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
            return {'fn': file_name, 'xyz_original': xyz_original,
                    'batch_offset': batch_offset,
                    'sup': superpoint, 'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch,
                    'v2p_index': v2p_index_batch}


