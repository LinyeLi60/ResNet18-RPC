
import torch.nn as nn
import torch


def _nms(heat, kernel=5):
    """

    :param heat:
    :param kernel: 改变kernel的大小可以获得更大的值
    :return:
    """
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()    # 取出那些最大值点
    return heat * keep    # 返回最大值点


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, K=100):
    """

    :param heat: 预测的热力图
    :param K: 取置信度最高的K个
    :return:
    """
    batch, cat, height, width = heat.size()

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    center_points = torch.cat([xs, ys], dim=2)    # shape=(batch, K, 2)
    detections = torch.cat([center_points, scores, clses], dim=2)
    return detections

