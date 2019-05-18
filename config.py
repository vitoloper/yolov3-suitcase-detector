import os
import json
opj = os.path.join

ROOT = '/home/giuseppe/image-processing/yolov3-suitcase-detector'
LOG_ROOT = opj(ROOT, 'logs')
CKPT_ROOT = opj(ROOT, 'checkpoints')

DATASET = 'coco'    # Assume this is the default dataset

def parse_names(path):
    """Parse names .json"""
    with open(path) as json_data:
        d = json.load(json_data)
        return d


def create_category_mapping(d):
    mapping = dict()
    for idx, id in enumerate(d):
        mapping[id] = idx
    return mapping


# datasets config
datasets = {
    'coco': {
        'num_classes': 1,
        'train_imgs': '/home/giuseppe/image-processing/suitcases/images',
        'val_imgs': '/home/giuseppe/image-processing/suitcases/images',
        'train_anno': '/home/giuseppe/image-processing/suitcases/instances_train.json',
        'val_anno': '/home/giuseppe/image-processing/suitcases/instances_val.json',
        'category_id_mapping': create_category_mapping([1]),
        'class_names': ['suitcase']
    },
    'voc': {
        'num_classes': 20,
        'train_imgs': '/media/data_2/VOCdevkit/voc_train.txt',
        'val_imgs': '/media/data_2/VOCdevkit/2007_test.txt',
        'class_names': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        'result_dir': opj(ROOT, 'metrics/voc/detections')
    },
    'linemod': {
        'num_classes': 1,
        'root': '/media/data_2/SIXDB/hinterstoisser/test/'
    }
}

# network config
# NOTE: we need a modified yolov3 cfg file since we have one class only
network = {
    'voc': {
        'cfg': opj(ROOT, 'lib/yolov3-voc.cfg')
    },
    'coco': {
        'cfg': opj(ROOT, 'cfg/yolov3-suitcase.cfg')  # This will be used on our dataset
    },
    'linemod': {
        'cfg': opj(ROOT, 'lib/yolov3-linemod.cfg')
    }
}

# evaluation config
evaluate = {
    'result_dir': opj(ROOT, 'assets/results')
}

colors = {}
