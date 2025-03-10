'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import os
import glob
import torch
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from plyfile import PlyData, PlyElement
import SharedArray as SA
import segmentator
# from tools.plt import write_ply

# ##if use Sharedmemory
use_shm_flag = True

# ###define the file path
SCANNET_DIR = './data/ScanNet/'
LABEL_MAP_FILE = './data/ScanNet/scannetv2-labels.combined.tsv'
OUTPUT_FOLDER = './datasets/ScanNetv2/npy/'
Dataset_DIR = './datasets/ScanNetv2/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LABEL_PLY_FOLDER = './datasets/ScanNetv2/label_ply/'
# os.makedirs(LABEL_PLY_FOLDER, exist_ok=True)

VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -100

NORMALIZED_CKASS_IDS_v2 = [IGNORE_INDEX for _ in range(1192)]
REVERSE_NORMALIZED_CKASS_IDS_v2 = [IGNORE_INDEX for _ in range(20)]
count_id = 0
for i, cls_id in enumerate(VALID_CLASS_IDS_20):
    NORMALIZED_CKASS_IDS_v2[cls_id] = count_id
    REVERSE_NORMALIZED_CKASS_IDS_v2[count_id] = cls_id
    count_id += 1
REVERSE_NORMALIZED_CKASS_IDS_v2_np = np.array(REVERSE_NORMALIZED_CKASS_IDS_v2)
labels_pd = pd.read_csv(LABEL_MAP_FILE, sep="\t", header=0)

# Map the raw category id to the point cloud
def point_indices_from_group(points, seg_indices, group, labels_pd, CLASS_IDs):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_ids = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    # Only store for the valid categories
    if label_id not in CLASS_IDs:
        label_id = 0

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))

    return points[point_IDs], point_IDs[0], label_id


# ### read XYZ RGB for each vertex. ( RGB values are in [-1,1])
def read_mesh_vertices_rgb(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
        xyz = vertices[:, :3] - vertices[:, :3].mean(0)
        rgb = vertices[:, 3:]/127.5 - 1

        faces = plydata['face'].data['vertex_indices']
    return xyz, rgb, faces, vertices[:, 3:]


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def f_test(fn):
    scan_name = fn.split('/')[-1]
    scan_name = scan_name[:12]
    output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
    print(scan_name)

    # ####get input
    xyz, rgb, faces, rgb_int = read_mesh_vertices_rgb(fn)

    # ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    normal_line_vertex = vertex_normal(xyz, face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(xyz.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()

    np.save(output_filename_prefix + '_coord.npy', xyz)
    np.save(output_filename_prefix + '_color.npy', rgb)
    np.save(output_filename_prefix + '_normal.npy', normal_line_vertex)
    np.save(output_filename_prefix + '_face.npy', face_npy)
    np.save(output_filename_prefix + '_superpoint.npy', superpoint)


def f(fn):
    # fn2 = fn[:-15] + '.txt'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    scan_name = fn.split('/')[-1]
    scan_name = scan_name[:12]
    output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
    # output_ply_prefix = os.path.join(LABEL_PLY_FOLDER, scan_name)
    print(scan_name)

    # ####get input
    xyz, rgb, faces, rgb_int = read_mesh_vertices_rgb(fn)

    # ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    normal_line_vertex = vertex_normal(xyz, face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(xyz.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()

    # Load segments file
    with open(fn3) as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])

    # Load Aggregations file
    with open(fn4) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation["segGroups"])

    # Generate new labels
    labelled_pc = np.ones((xyz.shape[0], 1))*-100
    instance_ids = np.ones((xyz.shape[0], 1))*-100
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(
            xyz, seg_indices, group, labels_pd, CLASS_IDS20
        )

        # labelled_pc[p_inds] = label_id
        labelled_pc[p_inds] = NORMALIZED_CKASS_IDS_v2[label_id]
        instance_ids[p_inds] = group["id"]

    labelled_pc = labelled_pc.astype(int).reshape(-1)
    instance_ids = instance_ids.astype(int).reshape(-1)
    # print(labelled_pc.shape)
    # write_ply(output_ply_prefix+'_label.ply', xyz, face_npy, labelled_pc, dataset='ScanNetv2')

    np.save(output_filename_prefix + '_coord.npy', xyz)
    np.save(output_filename_prefix + '_color.npy', rgb)
    np.save(output_filename_prefix + '_segment.npy', labelled_pc)
    np.save(output_filename_prefix + '_instance.npy', instance_ids)
    np.save(output_filename_prefix + '_normal.npy', normal_line_vertex)
    np.save(output_filename_prefix + '_face.npy', face_npy)
    np.save(output_filename_prefix + '_superpoint.npy', superpoint)

##################################################create shm###########################################

# #####shm create
def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def create_shm_train(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/v2_{}_normal".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            coord = np.load(npy_path + '{}_coord.npy'.format(fn))
            color = np.load(npy_path + '{}_color.npy'.format(fn))
            segment = np.load(npy_path + '{}_segment.npy'.format(fn))
            instance = np.load(npy_path + '{}_instance.npy'.format(fn))
            normal = np.load(npy_path + '{}_normal.npy'.format(fn))
            # ####write share memory
            sa_create("shm://v2_{}_coord".format(fn), coord)
            sa_create("shm://v2_{}_color".format(fn), color)
            sa_create("shm://v2_{}_segment".format(fn), segment)
            sa_create("shm://v2_{}_instance".format(fn), instance)
            sa_create("shm://v2_{}_normal".format(fn), normal)


def create_shm_val(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/v2_{}_normal".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            coord = np.load(npy_path + '{}_coord.npy'.format(fn))
            color = np.load(npy_path + '{}_color.npy'.format(fn))
            segment = np.load(npy_path + '{}_segment.npy'.format(fn))
            instance = np.load(npy_path + '{}_instance.npy'.format(fn))
            normal = np.load(npy_path + '{}_normal.npy'.format(fn))
            superpoint = np.load(npy_path + '{}_superpoint.npy'.format(fn))
            # ####write share memory
            sa_create("shm://v2_{}_coord".format(fn), coord)
            sa_create("shm://v2_{}_color".format(fn), color)
            sa_create("shm://v2_{}_segment".format(fn), segment)
            sa_create("shm://v2_{}_instance".format(fn), instance)
            sa_create("shm://v2_{}_superpoint".format(fn), superpoint)
            sa_create("shm://v2_{}_normal".format(fn), normal)

def create_shm_test(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/v2_{}_normal".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            coord = np.load(npy_path + '{}_coord.npy'.format(fn))
            color = np.load(npy_path + '{}_color.npy'.format(fn))
            normal = np.load(npy_path + '{}_normal.npy'.format(fn))
            superpoint = np.load(npy_path + '{}_superpoint.npy'.format(fn))
            # ####write share memory
            sa_create("shm://v2_{}_coord".format(fn), coord)
            sa_create("shm://v2_{}_color".format(fn), color)
            sa_create("shm://v2_{}_superpoint".format(fn), superpoint)
            sa_create("shm://v2_{}_normal".format(fn), normal)

# #############decode train val test set####################
train_files = glob.glob(SCANNET_DIR + 'train/*_vh_clean_2.ply')
val_files = glob.glob(SCANNET_DIR + 'val/*_vh_clean_2.ply')
test_files = glob.glob(SCANNET_DIR + 'test/*_vh_clean_2.ply')
train_files.sort(), val_files.sort(), test_files.sort()
p = mp.Pool(processes=mp.cpu_count())
p.map(f, train_files)
p.map(f, val_files)
p.map(f_test, test_files)
p.close()
p.join()


if use_shm_flag:
    train_list = np.loadtxt(Dataset_DIR + 'scannetv2_train.txt', dtype='str')
    val_list = np.loadtxt(Dataset_DIR + 'scannetv2_val.txt', dtype='str')
    test_list = np.loadtxt(Dataset_DIR + 'scannetv2_test.txt', dtype='str')
    create_shm_train(train_list, OUTPUT_FOLDER)
    create_shm_val(val_list, OUTPUT_FOLDER)
    create_shm_test(test_list, OUTPUT_FOLDER)


