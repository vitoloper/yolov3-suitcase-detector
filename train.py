from util import save_checkpoint, load_checkpoint
from dataset import prepare_train_dataset, prepare_val_dataset
from darknet import Darknet
# from model import YOLOv3
import config
import os
import sys
import time
import torch
import argparse
import warnings
from pprint import pprint
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm, trange
# from tensorboardX import SummaryWriter
# from torchvision import transforms, utils
import torch.optim.lr_scheduler as lr_scheduler
opj = os.path.join
warnings.filterwarnings("ignore")


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--reso', type=int, default=416,
                        help="Input image resolution")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--bs', type=int, default=8, help="Batch size")
    # parser.add_argument('--dataset', type=str, help="Dataset name",
    #                     choices=['voc', 'coco', 'linemod'])
    parser.add_argument('--ckpt', type=str, default='-1.-1',
                        help="Checkpoint name in format: `epoch.iteration`")
    parser.add_argument('--gpu', type=str, default='0', help="GPU id")
    parser.add_argument('--seq', type=str, help="LINEMOD sequence number")
    return parser.parse_args()

args = parse_arg()

# Load COCO configuration (config.DATASET is set to 'coco')
cfg = config.network[config.DATASET]['cfg']
# log_dir = opj(config.LOG_ROOT, get_current_time())
# writer = SummaryWriter(log_dir=log_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CUDA = torch.cuda.is_available()

def train(epoch, trainloader, yolo, optimizer):
    """Training wrapper
    Args
    - epoch: (int) training epoch
    - trainloader: (Dataloader) train data loader
    - yolo: (nn.Module) YOLOv3 model (Darknet backbone)
    - optimizer: (optim) optimizer

    """

    # Set network in training mode
    yolo.train()

    # tbar = tqdm(trainloader, ncols=80, ascii=True)
    # tbar.set_description('training')

    # Compute for each batch
    for batch_idx, (paths, inputs, targets) in enumerate(trainloader):
        print('batch_idx: {:d}'.format(batch_idx))
        print('epoch: {:d}'.format(epoch))
        print('len(trainloader): {:d}'.format(len(trainloader)))
        
        global_step = batch_idx + epoch * len(trainloader)
        # print(targets)  # Ground truth
        # forward + backward + optimize
        
        # Zeroes the gradient buffers of all parameters
        optimizer.zero_grad()
        
        # TODO: re-enable cuda
        # inputs = inputs.cuda()

        loss = yolo(inputs, targets, CUDA)
        
        pprint('[LOSS]')
        pprint(loss)
        print('global step: {0:6.0f}'.format(global_step))
        # log(writer, 'training loss', loss, global_step)
        
        # Backpropagation (backpropagate the error, the whole graph is differentiated w.r.t. the loss)
        loss['total'].backward()

        # Update the parameters
        optimizer.step()

        # tbar.set_postfix(loss=loss['total'])


# Main
if __name__ == '__main__':
    # Loading network
    # TODO: resume tensorboard

    print("[LOG] Loading network and data")

    # Create network
    yolo = Darknet(cfg)
    print("[LOG] Network created")
    
    # Set image resolution
    yolo.net_info["height"] = args.reso
    inp_dim = int(yolo.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    # Set starting epoch and iteration (default to -1.-1 means start with pretrained weights)
    start_epoch, start_iteration = args.ckpt.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        opj(config.CKPT_ROOT, config.DATASET),
        int(start_epoch),
        int(start_iteration)
    )

    # Load state dict
    yolo.load_state_dict(state_dict)

    print("[LOG] Checkpoint loaded ({:d}.{:d})".format(start_epoch, start_iteration))

    # Freeze the first 74 layers
    after_stop_layer = False
    stop_layer_name = 'module_list.75.conv_75.weight'
    for name, param in yolo.named_parameters():
        if name == stop_layer_name or after_stop_layer:
            param.requires_grad = True
            after_stop_layer = True
        else:
            param.requires_grad = False

    # Freeze layers:
    # Freeze layers
    # for name, param in yolo.named_parameters():
    #     if name not in layers_to_exclude:
    #         param.requires_grad = False

    # DEBUG: print trainable layers name
    print("[LOG] List of trainable parameters:")
    for name, param in yolo.named_parameters():
        if param.requires_grad:
            print(name)

    # Enable CUDA
    # yolo = yolo.cuda()

    # Prepare data
    train_img_datasets, train_dataloader = prepare_train_dataset(
        config.DATASET, args.reso, args.bs, seq=args.seq)
    val_img_datasets, val_dataloder = prepare_val_dataset(
        config.DATASET, args.reso, args.bs, seq=args.seq)
    print("[LOG] Model starts training from epoch %d iteration %d" %
          (start_epoch, start_iteration))
    print("[LOG] Number of training images:", len(train_img_datasets))
    print("[LOG] Number of validation images:", len(val_img_datasets))

    # Start training
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, yolo.parameters()),
                          lr=args.lr, momentum=0.8, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(start_epoch, start_epoch+100):
        print("\n[LOG] Epoch", epoch)
        
        # scheduler.step()

        # Call training wrapper
        train(epoch, train_dataloader, yolo, optimizer)

        # TODO: validate (by using validation set)

        scheduler.step()

        if (epoch == 10):
            save_checkpoint(opj(config.CKPT_ROOT, config.DATASET), epoch + 1, 0, {
                 'epoch': epoch + 1,
                 'iteration': 0,
                 'state_dict': yolo.state_dict()})

            print('Checkpoint saved, exiting.')
            sys.exit(0)
