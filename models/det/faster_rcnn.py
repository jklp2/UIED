import pdb
import sys

import numpy as np
import torch
import cv2
import os

from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from optimizer import get_optimizer
from scheduler import get_scheduler

# from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image,write_loss

import misc_utils as utils
import ipdb

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models import vgg16
from utils.map import eval_detection_voc
# from dataloader.coco import coco_90_to_80_classes
from tqdm import tqdm

def FasterRCNN_VGG():
    backbone = vgg16(pretrained=True).features
    backbone._modules.pop('30')  # 去掉最后一层Max_Pool层

    # for layer in range(10):  # 冻结conv3之前的层
    #     for p in backbone[layer].parameters():
    #         p.requires_grad = False

    backbone.out_channels = 512
    # backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes=opt.num_classes + 1)

    return model


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # self.detector = FasterRCNN_VGG()
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
        # print_network(self.detector)

        self.optimizer = get_optimizer("adam", self.detector)
        self.scheduler = get_scheduler("faster_rcnn", self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = "checkpoints/faster_rcnn"
        self.device = "cuda:0"

    def update(self, sample, *arg):
        """
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        labels = sample['det_labels']
        # ipdb.set_trace()
        
            # label += 1.  # effdet的label从1开始
        # ipdb.set_trace()

        image, bboxes, labels = sample['det_img'], sample['det_bboxes'], sample['det_labels']
        
        if len(bboxes[0]) == 0:  # 没有bbox，不更新参数
            return {}
        for label in labels:
            label += 1.
        image = image.to(self.device)
        bboxes = [bbox.to(self.device).float() for bbox in bboxes]
        labels = [label.to(self.device).float() for label in labels]
        image = list(im for im in image)

        b = len(bboxes)

        target = [{'boxes': bboxes[i], 'labels': labels[i].long()} for i in range(b)]
        """
            target['boxes'] = boxes
            target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd
        """
        loss_dict = self.detector(image, target)

        loss = sum(l for l in loss_dict.values())

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward(self, image):  # test
        conf_thresh = 0.5

        image = list(im for im in image)

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        with torch.no_grad():
            outputs = self.detector(image)

        for b in range(len(outputs)):  #
            output = outputs[b]
            boxes = output['boxes']
            labels = output['labels']
            scores = output['scores']
            boxes = boxes[scores > conf_thresh]
            labels = labels[scores > conf_thresh]
            labels = labels.detach().cpu().numpy()
            # for i in range(len(labels)):
            #     labels[i] = coco_90_to_80_classes(labels[i])

            labels = labels - 1
            scores = scores[scores > conf_thresh]

            batch_bboxes.append(boxes.detach().cpu().numpy())
            batch_labels.append(labels)
            batch_scores.append(scores.detach().cpu().numpy())
        import ipdb;ipdb.set_trace()
        return batch_bboxes, batch_labels, batch_scores
    def eval_mAP(self, dataloader, epoch, writer, logger=None, data_name='val'):
        # eval_yolo(self.detector, dataloader, epoch, writer, logger, dataname=data_name)
        pred_bboxes = []
        pred_labels = []
        pred_scores = []
        gt_bboxes = []
        gt_labels = []
        gt_difficults = []

        with torch.no_grad():
            i = 0
            pbar = tqdm(dataloader)
            for sample in pbar:
                pbar.set_description("det_eval")
                image = sample['det_img'].to(self.device)
                gt_bbox = sample['det_bboxes']
                labels = sample['det_labels']
                paths = sample['det_path']
                # ipdb.set_trace()

                batch_bboxes, batch_labels, batch_scores = self.forward(image)
                pred_bboxes.extend(batch_bboxes)
                pred_labels.extend(batch_labels)
                pred_scores.extend(batch_scores)

                for b in range(len(gt_bbox)):
                    gt_bboxes.append(gt_bbox[b].detach().cpu().numpy())
                    gt_labels.append(labels[b].int().detach().cpu().numpy())
                    gt_difficults.append(np.array([False] * len(gt_bbox[b])))

                if False:  # 可视化预测结果
                    img = tensor2im(image).copy()
                    for x1, y1, x2, y2 in gt_bbox[0]:
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt
                    num = len(batch_scores[0])
                    for n in range(num):
                        if batch_scores[0][n] > 0.05:
                            x1, y1, x2, y2 = batch_bboxes[0][n]
                            x1 = int(round(x1))
                            y1 = int(round(y1))
                            x2 = int(round(x2))
                            y2 = int(round(y2))
                            cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)  # 红色的是预测的
                    write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')
                i+=1

            result = []
            for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                AP = eval_detection_voc(
                    pred_bboxes,
                    pred_labels,
                    pred_scores,
                    gt_bboxes,
                    gt_labels,
                    gt_difficults=None,
                    iou_thresh=iou_thresh,
                    use_07_metric=False)

                APs = AP['ap']
                mAP = AP['map']
                result.append(mAP)
                if logger:
                    logger.info(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:opt.num_classes])}, mAP: {mAP}')
                print(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:3])}, mAP: {mAP}')
                write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)
            if logger:
                logger.info(
                    f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP75): {sum(result)/len(result)}')
            print(f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP75): {sum(result)/len(result)}')

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)
