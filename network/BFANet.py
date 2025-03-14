# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# File Name: BFANet.py
# IDE: PyCharm

import ocnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from ocnn.octree import Octree
from network.octformer import OctFormer, BFANet_SegHeader

import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d

# from PointTransv3 import PointTransformerV3 as ptv3


class BFANet(nn.Module):
    def __init__(
            self, cfg,
            channels: List[int] = [96, 192, 384, 384],
            num_blocks: List[int] = [2, 2, 18, 2],
            num_heads: List[int] = [6, 12, 24, 24],
            patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
            nempty: bool = True, stem_down: int = 2, head_up: int = 2,
            fpn_channel: int = 168, head_drop: List[float] = [0.0, 0.0], **kwargs):
        super().__init__()
        self.backbone = OctFormer(
            cfg.in_channels, channels, num_blocks, num_heads, patch_size, dilation,
            drop_path, nempty, stem_down)
        self.head = BFANet_SegHeader(
            cfg.sem_num, channels, fpn_channel, nempty, head_up, head_drop)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self,  octree: Octree, depth: int,
                query_pts: torch.Tensor, original_xyz, batch_id):
        octree_feature = ocnn.modules.InputFeature('NDFP', nempty=True)
        data = octree_feature(octree)
        features = self.backbone(data, octree, depth)
        sem_score_pred, mar_score_pred, sem_score_v2, margin_score_v2 = self.head(features, octree, query_pts)
        return sem_score_pred, mar_score_pred, sem_score_v2, margin_score_v2


class BFANet_mink(nn.Module):
    def __init__(self, cfg):
        super(BFANet_mink, self).__init__()
        # # config
        self.batch_size = cfg.batch_size
        self.sem_num = cfg.sem_num
        self.voxel_size = cfg.voxel_size

        # # ME-UNet
        self.MEUnet = unet3d(in_channels=6, out_channels=32, arch='MinkUNet34C')

        # ####sematic
        self.linear_semantic = nn.Sequential(
            ME.MinkowskiLinear(32, 32, bias=False),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(32, self.sem_num, bias=True)
        )

        # ####margin
        self.linear_margin = nn.Sequential(
            ME.MinkowskiLinear(32, 32, bias=False),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(32, 1, bias=True),
            ME.MinkowskiSigmoid()
        )

        # #####init weight
        self.weight_initialization()

        # #####frozen net to reduce room
        self.fix_module = []
        # self.fix_module = ['Unet_backbone', 'linear_semantic', 'linear_margin'] # #for train score net

        module_map = {'Unet_backbone': self.MEUnet,
                      'linear_semantic': self.linear_semantic
                     }
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False
        pass

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, feat_voxel, xyz_voxel, v2p_v1):
        # start_time = time.time()
        cuda_cur_device = torch.cuda.current_device()
        # #============================Unet V1===================================================================
        inputs_v1 = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        # ####backbone
        point_feat = self.MEUnet(inputs_v1)
        # ####two branches
        sem_pred_score_v = self.linear_semantic(point_feat)  # [V, sem_num] float32
        margin_pred_score_v = self.linear_margin(point_feat)  # [V, 3] float32

        # TODO add BFANet_SegHeader

        # ####change sparse tensor to torch tensor
        sem_pred_score = sem_pred_score_v.F  # from sparse tensor to torch tensor
        margin_pred_score = margin_pred_score_v.F  # from sparse tensor to torch tensor

        # #####Voxel to point
        sem_pred_score = sem_pred_score[v2p_v1, ...]  # [N, sem_num] float32
        margin_pred_score = margin_pred_score[v2p_v1, ...]  # [N, 3] float32

        return sem_pred_score, margin_pred_score, sem_pred_score, margin_pred_score



def model_fn(batch, model, epoch, cfg, task='train'):

    if cfg.backbone == "octformer":
        octree, points = batch['octree'].cuda(), batch['points'].cuda()
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        sem_score_pred, mar_score_pred, sem_score_pred_v2, mar_score_pred_v2 = model(octree, octree.depth, query_pts,
                                                                                     points.points, points.batch_id.view(-1))
        sem_label = points.labels

    elif cfg.backbone == 'mink':
        xyz_voxel = batch['xyz_voxel']
        feat_voxel = batch['feat_voxel']
        v2p_index = batch['v2p_index']
        sem_score_pred, mar_score_pred, sem_score_pred_v2, mar_score_pred_v2= model(feat_voxel, xyz_voxel, v2p_index)
        sem_label = batch['sem_label'].cuda()

    # # semantic loss_v1
    margin_label = batch['margin_label'].type(torch.float32).cuda()
    sem_weight = torch.ones(sem_label.shape[0]).cuda() + margin_label * 9.0
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction= 'none').cuda()
    semantic_loss = semantic_criterion(sem_score_pred, sem_label)
    semantic_loss = semantic_loss * sem_weight.detach()
    semantic_loss = semantic_loss.mean()
    valid = (sem_label != -100).float()

    # multi-classes dice loss
    semantic_labels_ = sem_label[sem_label != -100]
    semantic_scores_ = sem_score_pred[sem_label != -100]
    one_hot_labels = F.one_hot(semantic_labels_, num_classes=cfg.sem_num)
    semantic_scores_softmax = F.softmax(semantic_scores_, dim=-1)
    dice_loss = dice_loss_multi_classes(semantic_scores_softmax, one_hot_labels).mean()

    # margin loss
    margin_label = batch['margin_label'].type(torch.float32).cuda()
    weight_mask = (sem_label != -100)  # #margin label do not have -100 so we use sem_label
    margin_bce = nn.BCELoss(reduction='none', weight=weight_mask).cuda()
    margin_loss = margin_bce(mar_score_pred.view(-1), margin_label)
    margin_loss = margin_loss.mean()

    # # margin dice
    margin_dice_loss = binary_dice_loss(mar_score_pred.view(-1)[weight_mask], margin_label[weight_mask])

    # # ##### v2 loss
    semantic_loss_v2 = semantic_criterion(sem_score_pred_v2, sem_label)
    semantic_loss_v2 = semantic_loss_v2 * sem_weight.detach()
    semantic_loss_v2 = semantic_loss_v2.mean()

    semantic_scores_v2_ = sem_score_pred_v2[sem_label != -100]
    semantic_scores_softmax_v2 = F.softmax(semantic_scores_v2_, dim=-1)
    dice_loss_v2 = dice_loss_multi_classes(semantic_scores_softmax_v2, one_hot_labels).mean()

    margin_loss_v2 = margin_bce(mar_score_pred_v2.view(-1), margin_label)
    margin_loss_v2 = margin_loss_v2.mean()

    margin_dice_loss_v2 = binary_dice_loss(mar_score_pred_v2.view(-1)[weight_mask], margin_label[weight_mask])

    loss = semantic_loss + dice_loss + margin_loss + margin_dice_loss +\
           semantic_loss_v2 + dice_loss_v2 + margin_loss_v2 + margin_dice_loss_v2

    with torch.no_grad():
        pred = {}
        sem_pred_p = sem_score_pred_v2.max(1)[1]
        margin_pred = (mar_score_pred_v2.view(-1) > 0.45)
        pred['sem_pred'] = sem_pred_p
        pred['sem_pred_score'] = sem_score_pred_v2
        pred['margin_pred'] = margin_pred.int()

        visual_dict = {}
        visual_dict['loss'] = loss.item()
        visual_dict['semantic_loss'] = semantic_loss.item()
        visual_dict['Dice_loss'] = dice_loss.item()
        visual_dict['margin_loss'] = margin_loss.item()
        visual_dict['margin_dice_loss'] = margin_dice_loss.item()

        visual_dict['semantic_loss_v2'] = semantic_loss_v2.item()
        visual_dict['Dice_loss_v2'] = dice_loss_v2.item()
        visual_dict['margin_loss_v2'] = margin_loss_v2.item()
        visual_dict['margin_dice_loss_v2'] = margin_dice_loss_v2.item()

        meter_dict = {}
        meter_dict['loss'] = (loss.item(), valid.sum())
        meter_dict['semantic_loss'] = (semantic_loss.item(), valid.sum())
        meter_dict['Dice_loss'] = (dice_loss.item(), valid.sum())
        meter_dict['margin_loss'] = (margin_loss.item(), valid.sum())
        meter_dict['margin_dice_loss'] = (margin_dice_loss.item(), valid.sum())
        #
        meter_dict['semantic_loss_v2'] = (semantic_loss_v2.item(), valid.sum())
        meter_dict['Dice_loss_v2'] = (dice_loss_v2.item(), valid.sum())
        meter_dict['margin_loss_v2'] = (margin_loss_v2.item(), valid.sum())
        meter_dict['margin_dice_loss_v2'] = (margin_dice_loss_v2.item(), valid.sum())
    return loss, pred, visual_dict, meter_dict


def model_fn_eval(batch, model, epoch, cfg, task='train'):

    if cfg.backbone == "octformer":
        octree, points = batch['octree'].cuda(), batch['points'].cuda()
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        sem_score_pred, mar_score_pred, sem_score_pred_v2, mar_score_pred_v2 = model(octree, octree.depth, query_pts,
                                                                                     points.points, points.batch_id.view(-1))

    elif cfg.backbone == 'mink':
        xyz_voxel = batch['xyz_voxel']
        feat_voxel = batch['feat_voxel']
        v2p_index = batch['v2p_index']
        sem_score_pred, mar_score_pred, sem_score_pred_v2, mar_score_pred_v2= model(feat_voxel, xyz_voxel, v2p_index)

    with torch.no_grad():
        pred = {}
        sem_pred_p = sem_score_pred_v2.max(1)[1]
        margin_pred = (mar_score_pred_v2.view(-1) > 0.45)
        pred['sem_pred'] = sem_pred_p
        pred['sem_pred_score'] = sem_score_pred_v2
        pred['margin_pred'] = margin_pred.int()
    return pred


def dice_loss_multi_classes(input, target, epsilon= 1e-5, weight=None):
    r"""
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1. - per_channel_dice

    return loss

def binary_dice_loss(pred, target):
    smooth = 1.0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice

    return dice_loss




