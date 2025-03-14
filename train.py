# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# File Name: train.py
# IDE: PyCharm

import ocnn
import os, sys
import time
import random
import torch
import numpy as np
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import tools.log as log

from math import cos, pi
from tensorboardX import SummaryWriter
# from tools.plt import write_ply
from lib.BFANet_lib.torch_io import bfanet_ops
from tools.mIOU import intersectionAndUnionGPU, align_superpoint_label
from datasets.ScanNet200.scannet200_splits import HEAD_REORDER_ID, COMMON_REORDER_ID, TAIL_REORDER_ID

# from config.config_mink_ScanNetv2 import get_parser
# from config.config_mink_ScanNet200 import get_parser
from config.config_octformer_ScanNetv2 import get_parser
# from config.config_octformer_ScanNet200 import get_parser

def points2octree(points):
    octree = ocnn.octree.Octree(11, 2)
    octree.build_octree(points)
    return octree


def save_pred_ply(batch, sem_pred):
    output_dir = cfg.logpath + 'pred_ply/'
    os.makedirs(output_dir, exist_ok=True)
    file_name = batch['fn'][0]
    output_file_name = output_dir+file_name+'_pred.ply'

    xyz = batch['xyz_original']
    face = np.load('datasets/scannet200/npy/{}_face.npy'.format(file_name))
    write_ply(output_file_name, xyz, face, sem_pred.detach().cpu().numpy())


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    model.train()

    # #for log the run time and remain time
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()  # initialization
    am_dict = {}

    # #start train
    for i, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end_time)  # update time

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)  # adjust lr

        # ####get margin label
        xyz_original = batch['xyz_original']
        sem_label = batch['sem_label']
        batch_offset = batch['batch_offset']

        if cfg.backbone == "octformer":
            # ####contrust the octree on cuda
            points_list = batch['points']
            points_list = [pts.cuda() for pts in points_list]
            octrees = [points2octree(pts) for pts in points_list]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            points = ocnn.octree.merge_points(points_list)
            batch['points'] = points
            batch['octree'] = octree
            inbox_mask = batch['inbox_mask']
            margin_label = bfanet_ops.detect_margin(xyz_original, sem_label, batch_offset, 0.006)
            batch['margin_label'] = margin_label[inbox_mask]
        else:
            margin_label = bfanet_ops.detect_margin(xyz_original, sem_label, batch_offset, 0.06)
            batch['margin_label'] = margin_label

        # #loss, result, visual_dict , meter_dict (visual_dict: tensorboardX, meter_dict: average batch loss)
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch, cfg, task='train')

        # # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #average batch loss, time for print
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = log.AverageMeter()
            am_dict[k].update(v[0], v[1])

        current_iter = (epoch-1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            sys.stdout.write("epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f})  data_time: {:.2f}({:.2f}) "
                             "iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                             .format(epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val,
                                     am_dict['loss'].avg,
                                     batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
                                     remain_time=remain_time))
            if (i == len(train_loader) - 1): print()

    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info("epoch: {}/{}, train loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                          time.time() - start_time))
        # #write tensorboardX
        lr = optimizer.param_groups[0]['lr']
        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_train', am_dict[k].avg, epoch)
                writer.add_scalar('train/learning_rate', lr, epoch)

        # # save pretrained model
        pretrain_file = log.checkpoint_save(model, optimizer, cfg.logpath, epoch, cfg.save_freq)
        logger.info('Saving {}'.format(pretrain_file))
    pass


def eval_epoch(val_loader, model, model_fn, best_mIou, epoch):
    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_time = time.time()

        intersection_meter = log.AverageMeter()
        union_meter = log.AverageMeter()
        target_meter = log.AverageMeter()

        for i, batch in enumerate(val_loader):
            torch.cuda.empty_cache()

            # ####get margin label
            xyz_original = batch['xyz_original']
            sem_label = batch['sem_label']
            batch_offset = batch['batch_offset']

            if cfg.backbone == "octformer":
                # ####contrust the octree on cuda
                points_list = batch['points']
                points_list = [pts.cuda() for pts in points_list]
                octrees = [points2octree(pts) for pts in points_list]
                octree = ocnn.octree.merge_octrees(octrees)
                octree.construct_all_neigh()
                points = ocnn.octree.merge_points(points_list)
                batch['points'] = points
                batch['octree'] = octree
                inbox_mask = batch['inbox_mask']
                margin_label = bfanet_ops.detect_margin(xyz_original, sem_label, batch_offset, 0.006)
                batch['margin_label'] = margin_label[inbox_mask]
            else:
                margin_label = bfanet_ops.detect_margin(xyz_original, sem_label, batch_offset, 0.06)
                batch['margin_label'] = margin_label

            # ####validation
            loss, pred, visual_dict, meter_dict = model_fn(batch, model, epoch, cfg, task='eval')
            # #==========================================sem eval=========================================
            pred_score = pred['sem_pred_score'].view(3, -1, cfg.sem_num)
            pred_score = torch.mean(pred_score, dim=0)
            pred_sem = pred_score.max(1)[1]
            # pred_sem = pred['sem_pred']

            # save_pth = pred_score.detach().cpu()
            # os.makedirs(cfg.logpath + "{}/".format(epoch), exist_ok=True)
            # torch.save(save_pth, cfg.logpath + "{}/{}.pth".format(epoch, batch['fn'][0]))

            # #####superpoint label
            sup_id = batch['sup'].cuda()
            sup_sem, sup_score = align_superpoint_label(pred_sem, sup_id, num_label=cfg.sem_num, ignore_label=-100)
            sup_sem = sup_sem[sup_id]


            if cfg.backbone == "octformer":
                sem_label = batch['points'].labels.type(torch.int64).cuda()[:pred_sem.shape[0]]
            else:
                sem_label = batch['sem_label'].type(torch.int64).cuda()[:pred_sem.shape[0]]
            intersection, union, target = intersectionAndUnionGPU(sup_sem.detach().clone(), sem_label.detach().clone(), cfg.sem_num, -100)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            # save_pred_ply(batch, pred_sem)

            # #average batch loss, time for print
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = log.AverageMeter()
                am_dict[k].update(v[0], v[1])
            if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
                sys.stdout.write(
                    "\riter: {}/{} loss: {:.4f}({:.4f}) Accuracy {accuracy:.4f} ".format(i + 1, len(val_loader),
                                                                                         am_dict['loss'].val,
                                                                                         am_dict['loss'].avg,
                                                                                         accuracy=accuracy))
                if (i == len(val_loader) - 1): print()
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            logger.info("epoch: {}/{}, val loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                            time.time() - start_time))

            # #write tensorboardX
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)
        # #calculate ACC
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        if cfg.dataset == 'ScanNet200':
            head_mIou = np.mean(iou_class[HEAD_REORDER_ID])
            common_mIou = np.mean(iou_class[COMMON_REORDER_ID])
            tail_mIou = np.mean(iou_class[TAIL_REORDER_ID])
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            logger.info('mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.  '.
                        format(mIoU, mAcc, allAcc))
            logger.info('best_mIoU: {:.4f}.  '.format(best_mIou))
            # #write tensorboardX
            writer.add_scalar('val/mIOU_eval', mIoU, epoch)
            writer.add_scalar('val/mAcc_eval', mAcc, epoch)
            writer.add_scalar('val/allACC_eval', allAcc, epoch)
            if cfg.dataset == 'ScanNet200':
                writer.add_scalar('val/head_mIOU_eval', head_mIou, epoch)
                writer.add_scalar('val/common_mIou_eval', common_mIou, epoch)
                writer.add_scalar('val/tail_mIou_eval', tail_mIou, epoch)
            if epoch == cfg.epochs:  writer.close()
            if mIoU > best_mIou:
                best_mIou = mIoU*1.0
                # # save pretrained model
                torch.save({'model': model.state_dict()}, cfg.logpath + 'best_mIou.pth')
                logger.info('Saving best_mIou: {}'.format(cfg.logpath + 'best_mIou.pth'))
            return best_mIou


def Distributed_training(gpu, cfgs):
    global cfg
    global best_mIou
    cfg = cfgs
    best_mIou = 0.0
    cfg.local_rank = gpu
    # logger and summary write
    if cfg.local_rank == 0:
        # logger
        global logger
        from tools.log import get_logger
        logger = get_logger(cfg)
        logger.info(cfg)  # log config
        # summary writer
        global writer
        writer = SummaryWriter(cfg.logpath)
    cfg.rank = cfg.node_rank * cfg.gpu_per_node + gpu
    print('[PID {}] rank: {}  world_size: {}'.format(os.getpid(), cfg.rank, cfg.world_size))
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % cfg.tcp_port, world_size=cfg.world_size,
                            rank=cfg.rank)
    if cfg.local_rank == 0:
        logger.info(cfg)
    # #set cuda
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    if cfg.local_rank == 0:
        logger.info('cuda available: {}'.format(use_cuda))

    # #create model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')
    if cfg.backbone == "mink":
        from network.BFANet import BFANet_mink as net
    elif cfg.backbone == "octformer":
        from network.BFANet import BFANet as net
    elif cfg.backbone == "ptv3":
        from network.BFANet import BFANet as net

    from network.BFANet import model_fn
    model = net(cfg)
    model = model.to(gpu)
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],  find_unused_parameters=True)
    model._set_static_graph()
    if cfg.local_rank == 0:
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)
    # load dataset
    if cfg.dataset == 'ScanNet200':
        from datasets.ScanNet200.scannet200_splits import HEAD_REORDER_ID, COMMON_REORDER_ID, TAIL_REORDER_ID
        from datasets.ScanNet200.dataset_preprocess import Dataset
    elif cfg.dataset == 'ScanNetv2':
        from datasets.ScanNetv2.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info('Training samples: {}'.format(len(dataset.train_file_list)))
        logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, None, cfg.logpath, dist=cfg.dist, pretrain_file=cfg.pretrain,
                                                        gpu=gpu)
    if cfg.local_rank == 0:
        logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                    else 'Start from epoch {}'.format(start_epoch))

    # ####from 0 to cfg.epochs-1
    for epoch in range(start_epoch, cfg.epochs + 1):
        dataset.train_sampler.set_epoch(epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        # # #validation
        if cfg.validation and (epoch % 4 == 0 or epoch == cfg.epochs):
            dataset.val_sampler.set_epoch(epoch)
            best_mIou = eval_epoch(dataset.val_data_loader, model, model_fn, best_mIou, epoch)
    pass


def SingleCard_training(gpu, cfgs):
    global cfg
    cfg = cfgs
    cfg.local_rank = gpu
    # logger and summary write
    # logger
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    logger.info('=> creating model ...')
    from network.BFANet import BFANet as net
    from network.BFANet import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    model = net(cfg.in_channels, cfg.out_channels)
    model = model.to(gpu)

    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)
    # load dataset
    if cfg.dataset == 'ScanNet200':
        from datasets.ScanNet200.scannet200_splits import HEAD_REORDER_ID, COMMON_REORDER_ID, TAIL_REORDER_ID
        from datasets.ScanNet200.dataset_preprocess import Dataset
    elif cfg.dataset == 'ScanNetv2':
        from datasets.ScanNetv2.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)
    dataset.trainLoader()
    dataset.valLoader()
    logger.info('Training samples: {}'.format(len(dataset.train_file_list)))
    logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, None, cfg.logpath, pretrain_file=cfg.pretrain, gpu=gpu)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, cfg.epochs):
        eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
    pass


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    cfg = get_parser()
    # # fix seed for debug
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)

    # # Determine whether it is distributed training
    cfg.world_size = cfg.nodes * cfg.gpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    if cfg.dist:
        mp.spawn(Distributed_training, nprocs=cfg.gpu_per_node, args=(cfg,))
    else:
        print("the performance for single card is lower than multi-card, "
              "we do not suggest to only use one card for training")
        SingleCard_training(cfg.local_rank, cfg)