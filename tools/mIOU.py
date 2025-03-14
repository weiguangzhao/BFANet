# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/7/31  下午8:39
# File Name: mIOU.py
# IDE: PyCharm
import torch
import numpy as np
from scipy.sparse import coo_matrix

sem_class = {'floor': 0, 'wall': 1, 'cabinet': 2, 'bed': 3, 'chair': 4, 'sofa': 5, 'table': 6, 'door': 7,
             'window': 8, 'bookshelf': 9, 'picture': 10, 'counter': 11, 'desk': 12, 'curtain': 13,
             'refrigerator': 14, 'showercurtrain': 15, 'toilet': 16, 'sink': 17, 'bathtub': 18, 'otherfurniture': 19}

Ins_class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
             'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
             'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'otherfurniture': 17}


def intersectionAndUnionGPU(output, target, K, ignore_index=-100):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def align_superpoint_label(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint.cpu().numpy() # superpoint has been compression
    col = labels.cpu().numpy()
    col[col < 0] = num_label
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)  # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = torch.Tensor(label_map.max(1) / label_map.sum(axis=1)).to(labels.device) # [num_superpoint, num_label + 1]

    return label, label_scores

