# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# File Name: plt.py
# IDE: PyCharm

import glob
import os
import SharedArray as SA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from datasets.ScanNet200.scannet200_constants import VALID_CLASS_IDS_200,SCANNET_COLOR_MAP_200,VALID_CLASS_IDS_20,SCANNET_COLOR_MAP_20

SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])

# for sematic
COLOR20 = np.array(
    [[151, 223, 137], [174, 198, 232], [31, 120, 180], [255, 188, 120], [188, 189, 35],
     [140, 86, 74], [255, 152, 151], [213, 39, 40], [196, 176, 213], [148, 103, 188],
     [196, 156, 148], [23, 190, 208], [247, 183, 210], [218, 219, 141], [254, 127, 14],
     [158, 218, 229], [43, 160, 45], [112, 128, 144], [227, 119, 194], [82, 83, 163]])

# for instance
COLOR40 = np.array(
    [[88, 170, 108], [174, 105, 226], [78, 194, 83], [198, 62, 165], [133, 188, 52], [97, 101, 219], [190, 177, 52],
     [139, 65, 168], [75, 202, 137], [225, 66, 129],
     [68, 135, 42], [226, 116, 210], [146, 186, 98], [68, 105, 201], [219, 148, 53], [85, 142, 235], [212, 85, 42],
     [78, 176, 223], [221, 63, 77], [68, 195, 195],
     [175, 58, 119], [81, 175, 144], [184, 70, 74], [40, 116, 79], [184, 134, 219], [130, 137, 46], [110, 89, 164],
     [92, 135, 74], [220, 140, 190], [94, 103, 39],
     [144, 154, 219], [160, 86, 40], [67, 107, 165], [194, 170, 104], [162, 95, 150], [143, 110, 44], [146, 72, 105],
     [225, 142, 106], [162, 83, 86], [227, 124, 143]])

COLOR64 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=5)


def get_ptcloud_img(xyz, rgb, bbox=None, bbox_flag=False):
    fig = plt.figure(figsize=(12, 12))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax = Axes3D(fig)
    ax.view_init(90, -90)
    if bbox_flag == True:
        angle = np.zeros([bbox.shape[0]])
        bbox_cam = get_3d_box_batch(bbox[:, 3:6], angle, bbox[:, :3])
        bbox_cam = bbox_cam.transpose(0, 2, 1)
        for i in range(bbox_cam.shape[0]):
            draw_box(ax, bbox_cam[i, :, :], axes=[0, 1, 2], color=COLOR64[i] / 255.0)

    # plot point
    # max, min = np.max(xyz), np.min(xyz)
    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)
    ax.set_xbound(min_x, max_x)
    ax.set_ybound(min_y, max_y)
    ax.set_zbound(min_z, max_z)
    ax.scatter(x, y, z, zdir='z', c=rgb, marker='.', s=20)
    plt.show()
    pass


def plot_box(center, size, ax, color):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    color = color / 255.0
    ox, oy, oz = center
    l, w, h = size  # x_length,y_width,z_height

    x = np.linspace(ox - l / 2, ox + l / 2, num=2)
    y = np.linspace(oy - w / 2, oy + w / 2, num=2)
    z = np.linspace(oz - h / 2, oz + h / 2, num=2)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1) * (oy - w / 2)
    y12 = np.ones_like(x1) * (oy + w / 2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2) * (oz - h / 2)
    z22 = np.ones_like(x2) * (oz + h / 2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3) * (ox - l / 2)
    x32 = np.ones_like(y3) * (ox + l / 2)

    from mpl_toolkits.mplot3d import Axes3D
    # outside surface
    ax.plot_wireframe(x1, y11, z1, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color=color, rstride=1, cstride=1, alpha=0.6, linewidth=5.0)


def get_ptcloud_img_v2(xyz, rgb, bbox=None, bbox_flag=False, pic_name='IR511', save_flag=False, show_flag=False):
    fig = plt.figure(figsize=(12, 12))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax = Axes3D(fig)
    ax.view_init(90, -90)

    # plot boxes
    if bbox_flag == True:
        center = bbox[:, :3]
        size = bbox[:, 3:6]
        for box_num in range(np.shape(center)[0]):
            center_num = center[box_num, :]
            size_num = size[box_num, :]
            plot_box(center_num, size_num, ax, COLOR64[box_num])
            pass

    # plot point
    # max, min = np.max(xyz), np.min(xyz)
    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)
    ax.set_xbound(min_x, max_x)
    ax.set_ybound(min_y, max_y)
    ax.set_zbound(min_z, max_z)
    ax.axis(False)
    ax.scatter(x, y, z, zdir='z', c=rgb, marker='.', s=5)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    if save_flag:
        plt.savefig(pic_name)
    if show_flag:
        plt.show()
    plt.close()
    pass


def get_ptcloud_img_v3(xyz, sem, bbox=None, bbox_flag=False, pic_name='IR511', save_flag=False, show_flag=True,
                       task='sem'):


    fig = plt.figure(figsize=(12, 12))
    ax = Axes3D(fig)
    ax.view_init(90, -90)
    if task == 'sem':
        # #filter -100
        valid_index = np.argwhere(sem != -100)[:, 0]
        xyz = xyz[valid_index, :]
        sem = sem[valid_index]
        rgb = COLOR20[sem] / 255.0
    elif task == 'ins':
        # #filter -100
        sem[sem==-100] = -1
        # unan = np.argwhere(sem < 2)[:, 0]
        while(sem.max()>63):
            sem[sem > 63] = sem[sem > 63] - 63
        rgb = COLOR64[sem] / 255.0
    else:
        print('task only supports sem and ins')

    # plot boxes
    if bbox_flag == True:
        center = bbox[:, :3]
        size = bbox[:, 3:6]
        for box_num in range(np.shape(center)[0]):
            center_num = center[box_num, :]
            size_num = size[box_num, :]
            plot_box(center_num, size_num, ax, COLOR64[box_num])
            pass

    # plot point
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # max, min = np.max(xyz), np.min(xyz)
    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)
    ax.set_xbound(min_x, max_x)
    ax.set_ybound(min_y, max_y)
    ax.set_zbound(min_z, max_z)
    ax.scatter(x, y, z, zdir='z', c=rgb, marker='.', s=5)
    ax.axis(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    if save_flag:
        plt.savefig(pic_name)
    if show_flag:
        plt.show()
    plt.close()
    pass


def write_ply(file_path, xyz_data, face_data, label_data, dataset='ScanNet200'):
    with open(file_path, 'w') as ply_file:
        # 写入头部信息
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(xyz_data)))
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        ply_file.write("property ushort label\n")
        ply_file.write("element face {}\nproperty list uchar int vertex_indices\n".format(len(face_data)))
        ply_file.write("end_header\n")

        # 写入点数据
        # print(np.unique(label_data))
        for i in range(len(xyz_data)):
            if label_data[i]==-100:
                cur_rgb = (0.0,0.0,0.0)
                cur_label_id = 200
            else:
                cur_label_id = label_data[i]
                if dataset=="ScanNet200":
                    cur_id = label_data[i]
                    # if cur_id ==1: cur_id=2
                    # elif cur_id==2: cur_id=1
                    ex_id = VALID_CLASS_IDS_200[cur_id]
                    cur_rgb = SCANNET_COLOR_MAP_200[ex_id]
                elif dataset == "ScanNetv2":
                    ex_id = VALID_CLASS_IDS_20[label_data[i]]
                    cur_rgb = SCANNET_COLOR_MAP_20[ex_id]
                else:
                    print("please check the dataset name")
            ply_file.write("{} {} {} {} {} {} {}\n".format(xyz_data[i, 0], xyz_data[i, 1], xyz_data[i, 2],
                                                            int(cur_rgb[0]),  int(cur_rgb[1]), int(cur_rgb[2]),
                                                            int(cur_label_id)))
        # 写入面数据
        for face in face_data:
            ply_file.write("3 {} {} {}\n".format(face[0], face[1], face[2]))