from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
import os
opj = os.path.join

def load_classes(namesfile):
    """
    Load classes names from file

    """

    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    Takes a detection feature map and turns it into a 2D sensor.

    """
    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """

    # if box1.is_cuda == True:
    #     box1 = box1.cpu()
    # if box2.is_cuda == True:
    #    box2 = box2.cpu()

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def unique(tensor):
    """
    Get the classes present in a single image (since there can be multiple true detection of the same class)
    """

    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    Use NMS (non maximum suppression) to obtain the 'true' detections

    This function write_results outputs a tensor of shape D x 8.
    D is the 'true' detection in all of the images, each represented by a row.
    Each detection has 8 attributes: index of the image in the batch, 4 corner coords,
    objectness score, score of the class with maximum confidence, index of that class.

    """

    # For each of the bbox having an objectness score below a threshold
    # we set the value of it's attribute to zero.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # Convert (center_x, center_y, width, height) to (top-left-corner_x, top-left_corner_y, right-bottom-corner_x, right-bottom-corner_y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # NOTE: the number of 'true' detections in every image may be different.
    # Therefore, confidence thresholding and NMS has to be done for one image at once.
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
           #confidence thresholding 
           #NMS

        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of bbox rows set to zero (which have an object confidence less than threshold)
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            # No detection, continue
            continue

        # For PyTorch 0.4 compatibility
        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

        # Iterate over classes
        for cls in img_classes:
            # perform NMS

            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            # Perform NMS
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            # NOTE: write indicates if tensor has been initialized or not.
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
                
    # Check if there has been at least one detection, otherwise return 0.
    try:
        return output
    except:
        return 0

def letterbox_image(img, inp_dim):
    """
    Resize image with unchanged aspect ratio using padding

    """

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def save_checkpoint(checkpoint_dir, epoch, iteration, save_dict):
    """Save checkpoint to path
    Args
    - path: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint file
    - iteration: (int) iteration of checkpoint in one epoch
    - save_dict: (dict) saving parameters dict

    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    assert epoch == save_dict['epoch'], "[ERROR] epoch != save_dict's start_epoch"
    assert iteration == save_dict['iteration'], "[ERROR] iteration != save_dict's start_iteration"
    if os.path.isfile(path):
        print("[WARNING] Overwrite checkpoint in epoch %d, iteration %d" %
              (epoch, iteration))
    try:
        torch.save(save_dict, path)
    except Exception:
        raise Exception("[ERROR] Fail to save checkpoint")

    print("[LOG] Checkpoint %d.%d.ckpt saved" % (epoch, iteration))

def load_checkpoint(checkpoint_dir, epoch, iteration):
    """Load checkpoint from path
    Args
    - checkpoint_dir: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint
    - iteration: (int) iteration of checkpoint in one epoch
    Returns
    - start_epoch: (int)
    - start_iteration: (int)
    - state_dict: (dict) state of model
    
    """

    path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    if not os.path.isfile(path):
        raise Exception("Checkpoint in epoch %d doesn't exist" % epoch)

    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    start_iteration = checkpoint['iteration']

    assert epoch == start_epoch, "epoch != checkpoint's start_epoch"
    assert iteration == start_iteration, "iteration != checkpoint's start_iteration"
    return start_epoch, start_iteration, state_dict

def get_current_time():
    """Get current datetime
    Returns
    - time: (str) time in format "month-dd"
    
    """

    time = str(datetime.datetime.now())
    dhms = time.split('-')[-1].split('.')[0]
    day, hour, minute, _ = dhms.replace(' ', ':').split(':')
    month = calendar.month_name[int(time.split('-')[1])][:3]
    time = month + '.' + day
    return str(time)

def xywh2xyxy(bbox):
    """
    Coordinate conversion xywh -> xyxy
    
    """

    bbox_ = bbox.clone()
    if len(bbox_.size()) == 1:
        bbox_ = bbox_.unsqueeze(0)
    xc, yc = bbox_[..., 0], bbox_[..., 1]
    half_w, half_h = bbox_[..., 2] / 2, bbox_[..., 3] / 2
    bbox_[..., 0] = xc - half_w
    bbox_[..., 1] = yc - half_h
    bbox_[..., 2] = xc + 2 * half_w
    bbox_[..., 3] = yc + 2 * half_h
    return bbox_