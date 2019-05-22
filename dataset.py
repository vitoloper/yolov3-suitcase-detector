import config
import os
import time
import sys
import torch
import pickle
import numpy as np
# from tqdm import tqdm
from PIL import Image
from xml.etree import ElementTree
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

opj = os.path.join

# Seed imgaug
ia.seed(int(time.time()))

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform, reso):
        super().__init__(
            root=root,
            annFile=annFile,
            transform=transform,
        )
        self.class_map = config.datasets['coco']['category_id_mapping']
        self.reso = reso

    def __getitem__(self, index):
        """
        Returns
        - path: (str) Image filename
        - img: (Tensor)
        - annos: (Tensor) Annotation with size [100, 5]
            5 = [xc, yc, w, h, label]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        w, h = img.size
        target = coco.loadAnns(ann_ids)
        annos = torch.zeros(len(target), 5)

        # if self.reso is None, skip the whole image augmentation part.
        # Apply image augmentations before coordinate conversion, resizing and toTensor transform.
        if self.reso is not None:
            #Convert image to numpy array
            img = np.array(img)

            # Get all the annotations (bboxes)
            bboxes = []
            for i in range(len(target)):
                bbox = torch.Tensor(target[i]['bbox'])
                bbox_x = bbox[0]
                bbox_y = bbox[1]
                bbox_w = bbox[2]
                bbox_h = bbox[3]
                bboxes.append(BoundingBox(x1=bbox_x, y1=bbox_y, x2=bbox_x+bbox_w, y2=bbox_y+bbox_h))
            bbs = BoundingBoxesOnImage(bboxes, shape=img.shape)

            # NOTE: resize first, otherwise rotating will crop the image (since image is not square)
            # Rescale image and bounding boxes
            img = ia.imresize_single_image(img, (self.reso, self.reso))
            bbs = bbs.on(img)

            # Define augmentations
            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(1, iaa.Add((-50, 50))),    # Change saturation
                iaa.WithChannels(2, iaa.Add((-50,50))),      # Change intensity
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                iaa.Sometimes(0.5, iaa.Affine(rotate=90)),  # Rotate 90 deg. (0.5 probability)
                iaa.Affine(shear=(-2, 2)),  # Shear (-2 +2 degrees)
                iaa.Flipud(0.5),    # Flip up-down (with 0.5 probability)
                iaa.Fliplr(0.5)     # Flip left-right (with 0.5 probability)
            ])

            img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

            # # DEBUG: output image
            # image_after = bbs_aug.draw_on_image(img_aug)
            # # Convert numpy array to PIL image
            # image_after = Image.fromarray(image_after, 'RGB')
            # image_after.save('debug.png')
            # print(path)
            # print("DEBUG IMAGE SAVED")
            # print(bbox)
            # print(bbs_aug)
            # # sys.exit(0)

            # Overwrite original image
            img = img_aug
            

        for i in range(len(target)):
            # [x1, y1, w, h] => [xc, yc, w, h]

            # if self.reso is not None it means we have to update the bounding boxes coordinates
            if self.reso is not None:
                bbox = []
                bbox.append(int(bbs_aug.bounding_boxes[i].x1))
                bbox.append(int(bbs_aug.bounding_boxes[i].y1))
                bbox.append(int(bbs_aug.bounding_boxes[i].x2) - int(bbs_aug.bounding_boxes[i].x1))
                bbox.append(int(bbs_aug.bounding_boxes[i].y2) - int(bbs_aug.bounding_boxes[i].y1))
                bbox = torch.Tensor(bbox)
                w = self.reso
                h = self.reso
                # print(bbox)
                # print()
            else:
                bbox = torch.Tensor(target[i]['bbox'])

            annos[i, 0] = (bbox[0] + bbox[2] / 2) / w
            annos[i, 1] = (bbox[1] + bbox[3] / 2) / h
            annos[i, 2] = bbox[2] / w
            annos[i, 3] = bbox[3] / h
            annos[i, 4] = self.class_map[int(target[i]['category_id'])]

        #if self.transform is not None:
        #    img = self.transform(img)

        # ndarray -> Tensor conversion
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        if isinstance(img, torch.ByteTensor):
            img = img.float().div(255)

        return path, img, annos

    @staticmethod
    def collate_fn(batch):
        """Collate function for Coco DataLoader

        Returns
        - names: (tuple) each is a str of image filename
        - images: (Tensor) with size [bs, C, H, W]
        - annos: (tuple) each is a Tensor of annotations
        """
        names, images, annos = zip(*batch)
        images = default_collate(images)
        return names, images, annos


class VocDataset(torch.utils.data.dataset.Dataset):
    """
    Image datasets for PASCAL VOC

    Args
    - train_list: (str) full path to train list file
    """

    def __init__(self, train_list, transform):
        with open(train_list) as f:
            paths = f.readlines()
        self.img_paths = [x.strip() for x in paths]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_tensor = self.transform(Image.open(img_path))
        img_label_path = img_path.replace(
            'JPEGImages', 'labels').replace('.jpg', '.txt')
        img_anno = self.parse_label(img_label_path)
        return (img_path, img_tensor, img_anno)

    def __len__(self):
        return len(self.img_paths)

    def parse_label(self, label_path):
        """Parsing label

        Args
          label_path: (str) path to label file

        Returns
          img_anno: (Tensor) with size [#bbox, 5]
            offsets are scaled to (0,1) and in format [xc, yc, w, h, label]
        """
        bs = torch.Tensor(np.loadtxt(label_path))
        if len(bs.size()) == 1:  # only one object
            bs = bs.unsqueeze(0)

        img_anno = torch.zeros(bs.size())
        img_anno[:, :4] = bs[:, 1:]
        img_anno[:, 4] = bs[:, 0]

        return img_anno

    @staticmethod
    def collate_fn(batch):
        """Collate function for Voc DataLoader

        Returns
          paths: (tuple) each is a str of filepath to image
          images: (Tensor) with size [bs, C, H, W]
          annos: (tuple) each is a Tensor of annotations
        """
        names, images, annos = zip(*batch)
        images = default_collate(images)
        return names, images, annos


class LinemodDataset(torch.utils.data.dataset.Dataset):
    """
    Image datasets for LINEMOD
    """

    def __init__(self, root, seq, transform, is_train):
        """
        Args
        - root: (str) Root for LINEMOD test frames
        - seq: (str) Sequence for LINEMOD test frames
        - transform:
        - is_train: (bool)
        """
        all_lists = ['%04d' % x for x in range(
            len(os.listdir(opj(root, seq, 'rgb'))))]
        with open(opj(root, seq, 'train.txt')) as f:
            train_idxs = f.readlines()
        train_lists = [x.strip() for x in train_idxs]
        val_lists = list(set(all_lists) - set(train_lists))
        lists = train_lists if is_train else val_lists

        self.img_paths = [opj(root, seq, 'rgb/%s.png') %
                          x.strip() for x in lists]
        self.anno_paths = [opj(root, seq, 'annots/bbox/%s.npy') %
                           x.strip() for x in lists]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        w, h = img.size
        img_tensor = self.transform(img)
        img_anno = np.zeros((1, 5))
        img_anno[0, :4] = np.load(self.anno_paths[index])
        img_anno[0, 0] /= w
        img_anno[0, 1] /= h
        img_anno[0, 2] /= w
        img_anno[0, 3] /= h

        return (img_path, img_tensor, img_anno)

    def __len__(self):
        return len(self.img_paths)


def prepare_train_dataset(name, reso, batch_size, **kwargs):
    """Prepare dataset for training

    Args
    - name: (str) dataset name
    - reso: (int) training image resolution
    - batch_size: (int) default 32

    Returns
    - img_datasets: (CocoDataset) image datasets
    - trainloader: (Dataloader) dataloader for training
    """

    # We just want to transform the image into a tensor, since augmentations
    # are already performed in CocoDataset (only for training dataset)
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=reso, interpolation=3),
        # transforms.Resize(size=(reso, reso), interpolation=3),
        # transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    path = config.datasets[name]

    if name == 'coco':
        img_datasets = CocoDataset(
            root=path['train_imgs'],
            annFile=path['train_anno'],
            transform=None,     # Don't use any transformation
            reso=reso
        )
        dataloder = torch.utils.data.DataLoader(
            img_datasets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=CocoDataset.collate_fn
        )
    elif name == 'voc':
        img_datasets = VocDataset(
            train_list=path['train_imgs'],
            transform=transform
        )
        dataloder = torch.utils.data.DataLoader(
            img_datasets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=VocDataset.collate_fn
        )
    elif name == 'linemod':
        img_datasets = LinemodDataset(
            root=path['root'],
            seq=kwargs['seq'],
            transform=transform,
            is_train=True
        )
        dataloder = torch.utils.data.DataLoader(
            img_datasets, batch_size=batch_size, shuffle=True)
    else:
        raise NotImplementedError

    return img_datasets, dataloder


def prepare_val_dataset(name, reso, batch_size, **kwargs):
    """Prepare dataset for validation

    Args
    - name: (str) dataset name [tejani, hinter]
    - reso: (int) validation image resolution
    - batch_size: (int)

    Returns
    - img_datasets: (CocoDataset)
    - dataloader: (Dataloader)
    """
    transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ToTensor()
    ])

    path = config.datasets[name]

    if name == 'coco':
        img_datasets = CocoDataset(
            root=path['val_imgs'],
            annFile=path['val_anno'],
            transform=transform,
            reso=None
        )
        dataloder = torch.utils.data.DataLoader(
            img_datasets,
            batch_size=batch_size,
            collate_fn=CocoDataset.collate_fn
        )
    elif name == 'voc':
        img_datasets = VocDataset(
            train_list=path['val_imgs'], transform=transform)
        dataloder = torch.utils.data.DataLoader(
            img_datasets, batch_size=batch_size, collate_fn=VocDataset.collate_fn)
    elif name == 'linemod':
        img_datasets = LinemodDataset(
            root=path['root'],
            seq=kwargs['seq'],
            transform=transform,
            is_train=False
        )
        dataloder = torch.utils.data.DataLoader(
            img_datasets, batch_size=batch_size)
    else:
        raise NotImplementedError

    return img_datasets, dataloder
