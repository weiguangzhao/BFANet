# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# File Name: config.py
# IDE: PyCharm

import argparse

# config para
def get_parser():

    # # Train config
    parser = argparse.ArgumentParser(description='3D Semantic segmentation')
    parser.add_argument('--task', type=str, default='train', help='task: train or test')
    parser.add_argument('--manual_seed', type=int, default=22, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=401, help='Total epoch')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=4, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str, default='./log/ScanNetv2/octformer/epoch400_spherecrop12w_1e-3/', help='path to save logs')
    parser.add_argument('--cache', type=bool, default=True, help='Whether to use shm')
    parser.add_argument('--validation', type=bool, default=True, help='Whether to verify the validation set')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='ScanNetv2', help='datasets')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='voxel_size for voxelize')
    parser.add_argument('--sem_num', type=int, default=20, help='kinds of the sematic label')
    parser.add_argument('--crop_p', type=int, default=120000, help='point of scene for crop')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size for single GPU')
    parser.add_argument('--batch_size_v', type=int, default=1, help='batch_size for single GPU (validation)')
    parser.add_argument('--mixup', type=bool, default=True, help='mix up scene for data aug')

    # #Adjust learning rate
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--step_epoch', type=int, default=20, help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.5, help='Learning rate decay: lr = lr * multiplier')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for SGD')

    # #model parameter
    parser.add_argument('--backbone', type=str, default='octformer', help='backbone for training: mink, octformer, ptv3')
    parser.add_argument('--in_channels', type=int, default=10, help='in channels')
    parser.add_argument('--out_channels', type=int, default=32, help='out channels')

    # #Distributed training
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('-nr', '--node_rank', type=int, default=0, help='ranking within the nodes')
    parser.add_argument('--nodes', type=int, default=1, help='Number of distributed training nodes')
    parser.add_argument('--gpu_per_node', type=int, default=4, help='Number of GPUs per node')
    parser.add_argument('--sync_bn', type=bool, default=True, help='Whether to batch norm all gpu para')
    parser.add_argument('--tcp_port', type=int, default=12222, help='tcp port for Distributed training')

    args = parser.parse_args()
    return args