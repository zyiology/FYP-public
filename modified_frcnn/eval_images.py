import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
# import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import json
import math
import sys
import os
from PIL import Image
from custom_frcnn_dataset import CustomFRCNNAttentionDataset

import custom_faster_rcnn_attention as c_frcnn_attn
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import engine
import sys


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def eval_images(job_id=None, epoch_no=None):

    annotations_filepath = 'data/mapped_combined_annotations.json'
    num_classes = 2

    root = 'data/combined'
    if job_id is None:
        job_id = 215831

    if not isinstance(job_id, int):
        raise ValueError("Job ID should be an int")

    print(f"loading model from job {job_id}")

    # trainable_backbone_layers = 5 # range from 0 to 5 for resnet
    if epoch_no is None:
        epoch_no = 24

    target_attrib = "occupancy_group"
    use_model = True

    if not isinstance(epoch_no, int):
        raise ValueError("Epoch number should be an int")
        
    config_file = os.path.join('logs', str(job_id), 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)

    # use_attention = False
    # attention_per_attrib = False
    # custom_box_predictor = False
    # num_heads = 4
    
    # use_attribute_weights = True
    # use_resnet101_instead = True
    # use_pretrained_weights = False
    # use_reduced_features_for_attrib = True

    # val_ids_file = 'data/val_ids.txt'# f'logs/{job_id}/val_ids.txt'
    test_ids_file = 'data/test_ids.txt'
    checkpoint_file = f'checkpoints/{job_id}/epoch_{epoch_no}.pt'

    with open(test_ids_file, 'r') as f:
        test_ids = [int(id) for id in f.read().split(',')]
        
    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    exclude = None

    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        get_transform(train=False), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)

    # maybe i should raise error if no gpu
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if use_model:

        if config['use_resnet101_instead']:
            model = c_frcnn_attn.custom_fasterrcnn_resnet101_fpn(num_classes=num_classes,
                                                                attrib_mappings=attrib_mappings, 
                                                                trainable_backbone_layers=config['trainable_backbone_layers'],
                                                                attention_per_attrib=config['attention_per_attrib'],
                                                                #  pretrained=use_pretrained_weights,
                                                                use_attention=config['use_attention'],
                                                                use_reduced_features_for_attrib=config['use_reduced_features_for_attrib'],
                                                                custom_box_predictor=config['custom_box_predictor'],
                                                                num_heads=config['num_heads'],
                                                                parallel_backbone=config['parallel_backbone'])
        else:
            model = c_frcnn_attn.custom_fasterrcnn_resnet50_fpn(num_classes=num_classes, 
                                                                attrib_mappings=attrib_mappings, 
                                                                trainable_backbone_layers=config['trainable_backbone_layers'],
                                                                attention_per_attrib=config['attention_per_attrib'],
                                                                # pretrained=use_pretrained_weights,
                                                                use_attention=config['use_attention'],
                                                                use_reduced_features_for_attrib=config['use_reduced_features_for_attrib'],
                                                                custom_box_predictor=config['custom_box_predictor'],
                                                                num_heads=config['num_heads'])

        # model = c_frcnn_attn.custom_fasterrcnn_resnet50_fpn(num_classes=2, attrib_mappings=attrib_mappings, 
            # attention_per_attrib=attention_per_attrib, custom_box_predictor=custom_box_predictor)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    print("model loaded!")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    subfolder = os.path.join("eval_image", f"{job_id}")
    for index in test_ids:
        print(f"evaluating image_{index}...")
        c_frcnn_attn.eval_image(raw_dataset, epoch_no, device, save=True, index=index, target_attrib=target_attrib, 
                                show_attrib=True, model=model, subfolder=subfolder, eval_transform=get_transform(train=False),
                                score_threshold=0.7)
    

if __name__ == "__main__":

    print("running eval_images.py")

    server = True
    if server:
        eval_images()
    else:
        with open('.output.log', 'w') as f:
            sys.stdout = f
            eval_images()
