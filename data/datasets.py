import torch
import torch.utils.data as Data
import glob
import os
from torch.utils.data import Dataset,DataLoader
import ipdb
import albumentations as A
import xml.etree.ElementTree as ET
import random
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from utils.visual import get_summary_writer, visualize_boxes
from tqdm import tqdm
from mscv.summary import create_summary_writer, write_image, write_loss, write_meters_loss



classes = ['0B', '1B', '2B']

from os.path import join
def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    return img


def collate_fn(batch):
    target = {}
    b = len(batch)
    if 'det_img' in batch[0].keys():
        target['det_img'] = torch.stack([sample['det_img'] for sample in batch])
        target['det_bboxes'] = [sample['det_bboxes'] for sample in batch]
        target['det_labels'] = [sample['det_labels'] for sample in batch]
        target['det_path'] = [sample['det_path'] for sample in batch]
    if 'uieb_inp' in batch[0].keys():
        target['uieb_inp'] = torch.stack([sample['uieb_inp'] for sample in batch])
        target['uieb_ref'] = torch.stack([sample['uieb_ref'] for sample in batch])

    return target


class FRCNN(object):
    width = height = short_side = 600

    divisor = 32
    train_transform = A.Compose(  # FRCNN
        [
            A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),

            # A.RandomCrop(height=height, width=width, p=1.0),  # 600×600
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                                        val_shift_limit=0.3, p=0.95),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                            contrast_limit=0.3, p=0.95),
            ],p=1.0),

            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
    )

    # FRCNN
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


class uwdataset(Dataset):
    def __init__(self, det_path, uieb_path):
        divisor = 32
        short_side = 600
        self.det_transforms = A.Compose(  # FRCNN
            [
                A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
                A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor,
                              p=1.0),

                # A.RandomCrop(height=height, width=width, p=1.0),  # 600×600
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                                         val_shift_limit=0.3, p=0.95),
                    A.RandomBrightnessContrast(brightness_limit=0.3,
                                               contrast_limit=0.3, p=0.95),
                ], p=1.0),

                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            ),
        )
        self.transforms = A.Compose(
            [
                A.Resize(height=300, width=300, p=1.0),
                A.RandomCrop(height=256, width=256, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),  # TTA×8
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            additional_targets={'gt': 'image'}
        )
        with open(join(det_path, "ImageSets/Main/train.txt"), "r") as f:
            self.det_ids = f.readlines()
        self.det_ids = [x[:-1] for x in self.det_ids]
        self.det_img_paths = [join(join(det_path,"JPEGImages"),x+".jpg") for x in self.det_ids]
        self.det_anno_paths = [join(join(det_path,"Annotations"),x+".xml") for x in self.det_ids]
        self.det_infos = []
        for i in range(len(self.det_anno_paths)):
            path = self.det_anno_paths[i]
            info={}
            tree = ET.parse(path)
            root = tree.getroot()
            # ipdb.set_trace()
            for size in root.iter('size'):
                w = int(size[0].text)
                h = int(size[1].text)
            info['size'] = [h, w]  #[h, w]
            info['labels'] = []
            info['bboxes'] = []
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    info['bboxes'].append([int(bbox[i].text) for i in range(4)])
                info['labels'].append(classes.index(obj.find('name').text))
            info['path'] = self.det_img_paths[i]
            self.det_infos.append(info)
        self.uieb=[]
        with open(join(uieb_path,"train.txt"),"r") as f:
            lines = f.readlines()
        for line in lines:
            self.uieb.append([join(uieb_path,x) for x in line[:-1].split(" ")])
    def reset(self):
        random.shuffle(self.det_infos)
        random.shuffle(self.uieb)
    def __getitem__(self,idx):
        det_img = imread(self.det_infos[idx]['path'])
        det_labels = np.array(self.det_infos[idx]['labels'])
        det_bboxes = np.array(self.det_infos[idx]['bboxes'])
        sample = self.det_transforms(**{
                'image': det_img,
                'bboxes': det_bboxes,
                'labels': det_labels
        })        
        sample['bboxes'] = torch.Tensor(sample['bboxes']) 
        sample['labels'] = torch.Tensor(sample['labels'])
        uieb_inp = imread(self.uieb[idx][0])
        uieb_ref = imread(self.uieb[idx][1])
        sample2 = self.transforms(**{
            'gt':uieb_ref,
            'image':uieb_inp
        })
        return {"det_img": sample['image'],"det_labels": sample['labels'], "det_bboxes":sample['bboxes'],"det_path":self.det_infos[idx]['path'],"uieb_inp": sample2["image"], "uieb_ref":sample2["gt"]}

    def __len__(self):
        return min(len(self.det_infos),len(self.uieb))



class mmvaldataset(Dataset):
    def __init__(self, det_path):
        divisor = 32
        short_side = 600
        self.det_transforms = A.Compose(  # FRCNN
            [
                A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
                A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor,
                              pad_width_divisor=divisor, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            ),
        )
        with open(join(det_path, "ImageSets/Main/val.txt"), "r") as f:
            self.det_ids = f.readlines()
        self.det_ids = [x[:-1] for x in self.det_ids]
        self.det_img_paths = [join(join(det_path,"JPEGImages"),x+".jpg") for x in self.det_ids]
        self.det_anno_paths = [join(join(det_path,"Annotations"),x+".xml") for x in self.det_ids]
        self.det_infos = []
        for i in range(len(self.det_anno_paths)):
            path = self.det_anno_paths[i]
            info={}
            tree = ET.parse(path)
            root = tree.getroot()
            for size in root.iter('size'):
                w = int(size[0].text)
                h = int(size[1].text)
            info['size'] = [h, w]  #[h, w]
            info['labels'] = []
            info['bboxes'] = []
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    info['bboxes'].append([int(bbox[i].text) for i in range(4)])
                info['labels'].append(classes.index(obj.find('name').text))
            info['path'] = self.det_img_paths[i]
            self.det_infos.append(info)

    def reset(self):
        random.shuffle(self.det_infos)
    def __getitem__(self,idx):
        det_img = imread(self.det_infos[idx]['path'])
        det_labels = np.array(self.det_infos[idx]['labels'])
        det_bboxes = np.array(self.det_infos[idx]['bboxes'])
        sample = self.det_transforms(**{
                'image': det_img,
                'bboxes': det_bboxes,
                'labels': det_labels
        })
        sample['bboxes'] = torch.Tensor(sample['bboxes'])
        sample['labels'] = torch.Tensor(sample['labels'])
        return {"det_img": sample['image'], "det_labels": sample['labels'], "det_bboxes":sample['bboxes'],"det_path":self.det_infos[idx]['path']}
    def __len__(self):
        return len(self.det_infos)




class uiebvaldataset(Dataset):
    def __init__(self,  uieb_path, istrain = True):
        self.transforms = A.Compose(
            [
                A.Resize(height=300, width=300, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            additional_targets={'gt': 'image'}
        )
        self.uieb = []
        with open(join(uieb_path,"val.txt"),"r") as f:
            lines = f.readlines()
        for line in lines:
            self.uieb.append([join(uieb_path,x) for x in line[:-1].split(" ")])
    def reset(self):
        random.shuffle(self.uieb)
    def __getitem__(self,idx):
        uieb_inp = imread(self.uieb[idx][0])
        uieb_ref = imread(self.uieb[idx][1])
        sample2 = self.transforms(**{
            'gt':uieb_ref,
            'image':uieb_inp
        })
        return {"uieb_inp": sample2["image"], "uieb_ref":sample2["gt"]}

    def __len__(self):
        return len(self.uieb)

# class ruiedataset(Dataset):
#     def __init__(self,ruie_path):
#         super().__init__()
#         path_uccs = {}
#         path_uiqs = {}
#         path_utts = {}
#
#     def

if __name__=="__main__":
    dataset = uwdataset("/media/raid/underwater/chinamm2019uw/chinamm2019uw_train","/media/windows/c/datasets/underwater/UIEBD")
    dl = DataLoader(dataset,batch_size=2,num_workers=4,shuffle=True,collate_fn=collate_fn)
    dataset_val = uwdataset("/media/raid/underwater/chinamm2019uw/chinamm2019uw_train","/media/windows/c/datasets/underwater/UIEBD",istrain=False)
    dl_val = DataLoader(dataset_val,batch_size=1,num_workers=4,shuffle=True,collate_fn=collate_fn)

    # writer = get_summary_writer('logs','preview')
    
    # for i,data in tqdm(enumerate(dl)):
    #     image = data['det_img'][0].detach().cpu().numpy().transpose([1,2,0])
    #     image = (image.copy()*255).astype(np.uint8)
    #     bboxes = data['det_bboxes'][0].cpu().numpy()
    #     labels = data['det_labels'][0].cpu().numpy().astype(np.int32)
    #     visualize_boxes(image=image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(100, 101, size=[len(bboxes)])/100), class_labels=classes)
    #     write_image(writer, f'preview_mm2019/{i}', 'image', image, 0, 'HWC')

    #     # print(data["uieb_inp"].shape)

    # writer.flush()
    from models.det.faster_rcnn import Model
    det_model = Model().cuda()
    writer = get_summary_writer('logs','debug')
    steps = 0
    for epoch in range(200):
        dl.dataset.reset()
        it = 0
        pbar = tqdm(dl)
        for data in pbar:
            if it==0:
                print(data["det_path"])
            pbar.set_description('epoch: %i' % epoch)
            it+=1
            det_model.update(data)
            if it%10==0:
                write_meters_loss(writer, 'train',det_model.avg_meters, steps)
            steps+=1
            pbar.set_postfix(**(det_model.avg_meters.dic))
        det_model.eval()
        det_model.evaluate(dl_val,epoch,writer,None)
        det_model.train()


