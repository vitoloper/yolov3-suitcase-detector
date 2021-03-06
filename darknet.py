from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import *
from layers import DetectionLayer, DetectionLayerNoCuda


def get_test_input():
    """
    Get input test image

    """

    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# class DetectionLayer(nn.Module):
#     def __init__(self, anchors):
#         super(DetectionLayer, self).__init__()
#         self.anchors = anchors


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of unnecessary whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]            # Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()   # holds submodules in a list
    # We need to keep track of the number of filters in the layer on which the conv layer is being applied
    prev_filters = 3                # Initialize prev_filters to 3 since image has 3 filters corresponding to the RGB channels
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()    # nn.Sequential() class is used to sequentially execute a number of nn.Module() objects

        #check the type of block
        #create a new module for the block
        #append to module_list
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners=False)
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()    # empty layer
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                # we are concatenating maps
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

         #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(x["classes"])
            ignore_thresh = float(x["ignore_thresh"])   # Not used

            detection = DetectionLayerNoCuda(anchors, num_classes, int(net_info["height"]), ignore_thresh)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

# Test the creation of the layers
# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))


class Darknet(nn.Module):
    """
    Darknet class

    Args
        - cfgfile: path to network config file
    """

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, y_true, CUDA):
        modules = self.blocks[1:]   # first element is a net block which isn't part of the forward pass
        outputs = {}   #We cache the outputs for the route layer
        # detections = torch.Tensor().cuda()  # detection results
        detections = torch.Tensor()  # detection results
        loss = dict()

        # This flag is used to indicate whether we have encountered the first detection or not.
        # If 0: the collector hasn't been initialized.
        # If 1: the collector has been initialized and we can concatenate our detection maps to it.
        write = 0

        # iterate over modules
        for i, module in enumerate(modules):        
            module_type = (module["type"])

            # if module is convolutional or upsample
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module_type == 'yolo':   
                if self.training == True:
                    loss_part = self.module_list[i][0](x, y_true)
                    for key, value in loss_part.items():
                        value = value
                        loss[key] = loss[key] + \
                            value if key in loss.keys() else value
                        loss['total'] = loss['total'] + \
                            value if 'total' in loss.keys() else value
                else:
                    x = self.module_list[i][0](x)
                    detections = x if len(detections.size()) == 1 else torch.cat(
                        (detections, x), 1)

                outputs[i] = outputs[i-1]  # skip

        if self.training == True:
            return loss
        else:
            return detections

    def load_weights(self, weightfile, cutoff=None):
        """
        Load the weights

        """

        # Open weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        # Read weights from file
        weights = np.fromfile(fp, dtype = np.float32)

        # Cutoff
        if cutoff is not None:
            cutoff = cutoff + 1     # layer[i] = blocks[i + 1] because of [net] block
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #print(module_type)

            if cutoff is not None and (i+1) == cutoff:
                print("Stop before {:s} layer (No. {:d})".format( module_type, i))
                break
    
            #If module_type is convolutional load weights
            #Otherwise ignore.

            # print(self.module_list[i])
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                # Different weights loading mode if conv block has batch_normalize
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
 
                # Finally load the convolutional layer's weights.
                #Let us load the weights for the Convolutional layers
                # numel(): calculate number of elements in a torch tensor
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# NOTE:  the following lines are for testing purposes only

# model = Darknet("cfg/yolov3.cfg")
# model.load_weights("yolov3.weights")

# inp = get_test_input();
# pred = model(inp, torch.cuda.is_available())

# We should have a tensor of size torch.Size([1, 10647, 85])
# first dimension is batch size (just one test image)
# each row in the matrix 10647x85 is a bounding box
# (85 because we have 4 bbox attributes, 1 objectness score and 80 class scores)

# print(pred)
# print(pred.size())
