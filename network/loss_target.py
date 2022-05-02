import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


def bce_loss(y_true, y_pred_logits):
    # y_pred_prob = F.sigmoid(y_pred_logits)
    # y_true_f = y_true.view(-1)
    # y_true_f = y_true
    y_pred_logits = y_pred_logits.view(-1)
    y_pred_prob_f = y_pred_logits.clamp(min=1e-7, max=1 - 1e-7)
    return -(y_true * y_pred_prob_f.log() + (1. - y_true) * (1 - y_pred_prob_f).log()).mean()


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class Loss_target(nn.Module):
    def __init__(self, weight_angle=10):
        super(Loss_target, self).__init__()
        self.weight_angle = weight_angle
        self.bce = bce_loss

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map,pre_domain):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        classify_loss = get_dice_loss(gt_score, pred_score*(1-ignored_map))

        gt_doamin = torch.Tensor([[[1.]]]).to(torch.device("cuda"))
        doamin_loss = self.bce(gt_doamin, pre_domain)

        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        # print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
        return geo_loss, classify_loss, doamin_loss
