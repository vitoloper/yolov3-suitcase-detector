import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

import config
from util import bbox_iou, xywh2xyxy


class EmptyLayer(nn.Module):
    """Empty layer for shortcut connection"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """Detection layer

    Args
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int) # classes
      reso: (int) original image resolution
      ignore_thresh: (float)
    """

    def __init__(self, anchors, num_classes, reso, ignore_thresh):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.reso = reso
        self.ignore_thresh = ignore_thresh

    def forward(self, x, y_true=None):
        """
        Transform feature map into 2-D tensor. Transformation includes
        1. Re-organize tensor to make each row correspond to a bbox
        2. Transform center coordinates
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        3. Transform width and height
        bw = pw * exp(tw)
        bh = ph * exp(th)
        4. Activation
        @Args
        x: (Tensor) feature map with size [bs, (5+nC)*nA, gs, gs]
            5 => [4 offsets (xc, yc, w, h), objectness]
        @Returns
        detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
        """
        bs, _, gs, _ = x.size()
        stride = self.reso // gs  # no pooling used, stride is the only downsample
        num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
        nA = len(self.anchors)
        scaled_anchors = torch.Tensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]).cuda()

        # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, num_attrs, gs, gs).permute(
            0, 1, 3, 4, 2).contiguous()

        pred = torch.Tensor(bs, nA, gs, gs, num_attrs).cuda()

        pred_tx = torch.sigmoid(x[..., 0]).cuda()
        pred_ty = torch.sigmoid(x[..., 1]).cuda()
        pred_tw = x[..., 2].cuda()
        pred_th = x[..., 3].cuda()
        pred_conf = torch.sigmoid(x[..., 4]).cuda()
        if self.training == True:
            pred_cls = x[..., 5:].cuda()  # softmax in cross entropy
        else:
            pred_cls = F.softmax(x[..., 5:], dim=-1).cuda()  # class

        grid_x = torch.arange(gs).repeat(gs, 1).view(
            [1, 1, gs, gs]).float().cuda()
        grid_y = torch.arange(gs).repeat(gs, 1).t().view(
            [1, 1, gs, gs]).float().cuda()
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred[..., 0] = pred_tx + grid_x
        pred[..., 1] = pred_ty + grid_y
        pred[..., 2] = torch.exp(pred_tw) * anchor_w
        pred[..., 3] = torch.exp(pred_th) * anchor_h
        pred[..., 4] = pred_conf
        pred[..., 5:] = pred_cls

        if not self.training:
            pred[..., :4] *= stride
            return pred.view(bs, -1, num_attrs)
        else:
            gt_tx = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_ty = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_tw = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_th = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_conf = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_cls = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()

            obj_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            for idx in range(bs):
                for y_true_one in y_true[idx]:
                    y_true_one = y_true_one.cuda()
                    gt_bbox = y_true_one[:4] * gs
                    gt_cls_label = int(y_true_one[4])

                    gt_xc, gt_yc, gt_w, gt_h = gt_bbox[0:4]
                    gt_i = gt_xc.long().cuda()
                    gt_j = gt_yc.long().cuda()

                    pred_bbox = pred[idx, :, gt_j, gt_i, :4]
                    ious = bbox_iou(xywh2xyxy(pred_bbox), xywh2xyxy(gt_bbox))
                    best_iou, best_a = torch.max(ious, 0)

                    w, h = scaled_anchors[best_a]
                    gt_tw[idx, best_a, gt_j, gt_i] = torch.log(gt_w / w)
                    gt_th[idx, best_a, gt_j, gt_i] = torch.log(gt_h / h)
                    gt_tx[idx, best_a, gt_j, gt_i] = gt_xc - gt_i.float()
                    gt_ty[idx, best_a, gt_j, gt_i] = gt_yc - gt_j.float()
                    gt_conf[idx, best_a, gt_j, gt_i] = best_iou
                    gt_cls[idx, best_a, gt_j, gt_i] = gt_cls_label

                    obj_mask[idx, best_a, gt_j, gt_i] = 1

            MSELoss = nn.MSELoss(reduction='sum')
            BCELoss = nn.BCELoss(reduction='sum')
            CELoss = nn.CrossEntropyLoss(reduction='sum')

            loss = dict()
            loss['x'] = MSELoss(pred_tx * obj_mask, gt_tx * obj_mask)
            loss['y'] = MSELoss(pred_ty * obj_mask, gt_ty * obj_mask)
            loss['w'] = MSELoss(pred_tw * obj_mask, gt_tw * obj_mask)
            loss['h'] = MSELoss(pred_th * obj_mask, gt_th * obj_mask)
            # loss['cls'] = BCELoss(pred_cls * obj_mask, cls_mask * obj_mask)

            loss['cls'] = CELoss((pred_cls * obj_mask.unsqueeze(-1)).view(-1, self.num_classes),
                                 (gt_cls * obj_mask).view(-1).long())
            loss['conf'] = MSELoss(pred_conf * obj_mask * 5, gt_conf * obj_mask * 5) + \
                MSELoss(pred_conf * (1 - obj_mask), pred_conf * (1 - obj_mask))

            pprint(loss)

            return loss


class DetectionLayerNoCuda(nn.Module):
    """Detection layer

    Args
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int) # classes
      reso: (int) original image resolution
      ignore_thresh: (float)
    """

    def __init__(self, anchors, num_classes, reso, ignore_thresh):
        super(DetectionLayerNoCuda, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.reso = reso
        self.ignore_thresh = ignore_thresh

    def forward(self, x, y_true=None):
        """
        Transform feature map into 2-D tensor. Transformation includes
        1. Re-organize tensor to make each row correspond to a bbox
        2. Transform center coordinates
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        3. Transform width and height
        bw = pw * exp(tw)
        bh = ph * exp(th)
        4. Activation
        @Args
        x: (Tensor) feature map with size [bs, (5+nC)*nA, gs, gs]
            5 => [4 offsets (xc, yc, w, h), objectness]
        @Returns
        detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
        """
        bs, _, gs, _ = x.size()
        stride = self.reso // gs  # no pooling used, stride is the only downsample
        num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
        nA = len(self.anchors)
        scaled_anchors = torch.Tensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])

        # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, num_attrs, gs, gs).permute(
            0, 1, 3, 4, 2).contiguous()

        pred = torch.Tensor(bs, nA, gs, gs, num_attrs)

        pred_tx = torch.sigmoid(x[..., 0])
        pred_ty = torch.sigmoid(x[..., 1])
        pred_tw = x[..., 2]
        pred_th = x[..., 3]
        pred_conf = torch.sigmoid(x[..., 4])
        if self.training == True:
            pred_cls = x[..., 5:]  # softmax in cross entropy
        else:
            pred_cls = F.softmax(x[..., 5:], dim=-1)  # class

        grid_x = torch.arange(gs).repeat(gs, 1).view(
            [1, 1, gs, gs]).float()
        grid_y = torch.arange(gs).repeat(gs, 1).t().view(
            [1, 1, gs, gs]).float()
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred[..., 0] = pred_tx + grid_x
        pred[..., 1] = pred_ty + grid_y
        pred[..., 2] = torch.exp(pred_tw) * anchor_w
        pred[..., 3] = torch.exp(pred_th) * anchor_h
        pred[..., 4] = pred_conf
        pred[..., 5:] = pred_cls

        if not self.training:
            pred[..., :4] *= stride
            return pred.view(bs, -1, num_attrs)
        else:
            gt_tx = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            gt_ty = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            gt_tw = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            gt_th = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            gt_conf = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            gt_cls = torch.zeros(bs, nA, gs, gs, requires_grad=False)

            obj_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False)
            for idx in range(bs):
                for y_true_one in y_true[idx]:
                    y_true_one = y_true_one
                    gt_bbox = y_true_one[:4] * gs
                    gt_cls_label = int(y_true_one[4])

                    gt_xc, gt_yc, gt_w, gt_h = gt_bbox[0:4]
                    gt_i = gt_xc.long()
                    gt_j = gt_yc.long()

                    pred_bbox = pred[idx, :, gt_j, gt_i, :4]
                    ious = bbox_iou(xywh2xyxy(pred_bbox), xywh2xyxy(gt_bbox))
                    best_iou, best_a = torch.max(ious, 0)

                    w, h = scaled_anchors[best_a]
                    gt_tw[idx, best_a, gt_j, gt_i] = torch.log(gt_w / w)
                    gt_th[idx, best_a, gt_j, gt_i] = torch.log(gt_h / h)
                    gt_tx[idx, best_a, gt_j, gt_i] = gt_xc - gt_i.float()
                    gt_ty[idx, best_a, gt_j, gt_i] = gt_yc - gt_j.float()
                    gt_conf[idx, best_a, gt_j, gt_i] = best_iou
                    gt_cls[idx, best_a, gt_j, gt_i] = gt_cls_label

                    obj_mask[idx, best_a, gt_j, gt_i] = 1

            MSELoss = nn.MSELoss(reduction='sum')
            BCELoss = nn.BCELoss(reduction='sum')
            CELoss = nn.CrossEntropyLoss(reduction='sum')

            loss = dict()
            loss['x'] = MSELoss(pred_tx * obj_mask, gt_tx * obj_mask)
            loss['y'] = MSELoss(pred_ty * obj_mask, gt_ty * obj_mask)
            loss['w'] = MSELoss(pred_tw * obj_mask, gt_tw * obj_mask)
            loss['h'] = MSELoss(pred_th * obj_mask, gt_th * obj_mask)
            # loss['cls'] = BCELoss(pred_cls * obj_mask, cls_mask * obj_mask)

            loss['cls'] = CELoss((pred_cls * obj_mask.unsqueeze(-1)).view(-1, self.num_classes),
                                 (gt_cls * obj_mask).view(-1).long())
            loss['conf'] = MSELoss(pred_conf * obj_mask * 5, gt_conf * obj_mask * 5) + \
                MSELoss(pred_conf * (1 - obj_mask), pred_conf * (1 - obj_mask))

            pprint(loss)

            return loss
