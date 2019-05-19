import os
import config
import argparse
# from model import YOLOv3
from darknet import Darknet
from util import save_checkpoint
import sys
opj = os.path.join

def parse_arg():
    parser = argparse.ArgumentParser("Transfer .weights to PyTorch checkpoint")
    parser.add_argument('--weights', default='yolov3.weights',
                        type=str, help=".weights file name (stored in checkpoint/darknet)")
    parser.add_argument('--cutoff', default=None,
                        type=int, help="Layer cutoff value")
    return parser.parse_args()

# Main
if __name__ == '__main__':
    args = parse_arg()
    weight_path = opj(args.weights)
    cutoff = args.cutoff

    print("[LOG] Loading weights from", weight_path)
    print("[LOG] Cutoff value:", cutoff)
    model = Darknet(config.network[config.DATASET]['cfg'])

    model.load_weights(weight_path, cutoff=cutoff)

    # Save as a checkpoint file
    # NOTE: we set epoch and iteration to -1 because we aren't saving a training checkpoint.
    save_checkpoint(opj(config.CKPT_ROOT, config.DATASET), -1, -1, {
        'epoch': -1,
        'iteration': -1,
        'state_dict': model.state_dict(),
})
