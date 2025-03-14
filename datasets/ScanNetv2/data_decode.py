'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import os
import glob
import torch
import json
import plyfile
import numpy as np
import multiprocessing as mp
import SharedArray as SA

import segmentator


# ##if use cuda to decode the normal
use_shm_flag = True

# ###define the file path
SCANNET_DIR = './data/ScanNet/'
LABEL_MAP_FILE = './data/ScanNet/scannetv2-labels.combined.tsv'
OUTPUT_FOLDER = './datasets/ScanNetv2/npy/'
Dataset_DIR = './datasets/ScanNetv2/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']


def get_raw2scannetv2_label_map():
    lines = [line.rstrip() for line in open(LABEL_MAP_FILE)]
    lines_0 = lines[0].split('\t')
    print(lines_0)
    print(len(lines))
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        if (elements[1] != elements[2]):
            print('{}: {} {}'.format(i, elements[1], elements[2]))
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

g_raw2scannetv2 = get_raw2scannetv2_label_map()

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
        rgb = vertices[:, 3:]

        faces = plydata['face'].data['vertex_indices']
    return xyz, rgb, faces


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
    xyz, rgb, faces = read_mesh_vertices_rgb(fn)

    # ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    normal_line_vertex = vertex_normal(xyz, face_npy)
    # if use_cuda_flag == False:
    #     normal, areas, vertex_to_face = surface_normal_area(faces, xyz)
    #     normal_line_vertex = vertex_normal(vertex_to_face, normal, areas)
    # if use_cuda_flag == True:
    #     normal_line_vertex = get_normal_line(xyz, face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(xyz.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()
    np.save(output_filename_prefix + '_coord.npy', xyz)
    np.save(output_filename_prefix + '_color.npy', rgb)
    np.save(output_filename_prefix + '_normal.npy', normal_line_vertex)
    np.save(output_filename_prefix + '_face.npy', face_npy)
    np.save(output_filename_prefix + '_superpoint.npy', superpoint)


def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    scan_name = fn.split('/')[-1]
    scan_name = scan_name[:12]
    output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
    print(scan_name)

    # ####get input
    xyz, rgb, faces = read_mesh_vertices_rgb(fn)

    # ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    normal_line_vertex = vertex_normal(xyz, face_npy)
    # if use_cuda_flag == False:
    #     normal, areas, vertex_to_face = surface_normal_area(faces, xyz)
    #     normal_line_vertex = vertex_normal(vertex_to_face, normal, areas)
    # if use_cuda_flag == True:
    #     normal_line_vertex = get_normal_line(xyz, face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(xyz.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()

    # ###get label
    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            if g_raw2scannetv2[x['label']] != 'wall' and g_raw2scannetv2[x['label']] != 'floor':
                instance_segids.append(x['segments'])
                labels.append(x['label'])
                assert(x['label'] in g_raw2scannetv2.keys())
    if(scan_name == 'scene0217_00' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(sem_labels[pointids])) == 1)

    np.save(output_filename_prefix + '_coord.npy', xyz)
    np.save(output_filename_prefix + '_color.npy', rgb)
    np.save(output_filename_prefix + '_segment.npy', sem_labels)
    np.save(output_filename_prefix + '_instance.npy', instance_labels)
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

#############decode train val test set####################
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


