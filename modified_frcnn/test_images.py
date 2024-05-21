### SCRIPT TO TEST IMAGES USING A TRAINED MODEL ###
# Job ID and epoch number should be provided as arguments when calling this script

import torch
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
import json
import sys
import os

# from local files
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
import custom_faster_rcnn_attention as c_frcnn_attn
import utils
import sys


def get_transform():
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def test_images(job_id=None, epoch_no=None):

    # location of annotations file and dataset root folder
    annotations_filepath = 'data/mapped_combined_annotations.json'
    root = 'data/combined'

    # validate inputs
    if job_id is None:
        raise ValueError("need to provide Job ID to load corresponding checkpoint")

    if not isinstance(job_id, int):
        raise ValueError("Job ID should be an int")

    if epoch_no is None:
        raise ValueError("need to provide Epoch number to load corresponding checkpoint")

    if not isinstance(epoch_no, int):
        raise ValueError("Epoch number should be an int")
    
    print(f"loading model from job {job_id}, epoch {epoch_no}")
        
    # load config file for given job
    config_file = os.path.join('logs', str(job_id), 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)

    # load test IDs from file
    test_ids_file = 'data/test_ids.txt'
    with open(test_ids_file, 'r') as f:
        test_ids = [int(id) for id in f.read().split(',')]
        
    # load attribute mappings
    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        get_transform(train=False), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=None)

    test_dataset = torch.utils.data.Subset(raw_dataset, test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    model = c_frcnn_attn.custom_fasterrcnn_resnet_fpn(num_classes=config['num_classes'],
                                                      attrib_mappings=attrib_mappings, 
                                                      trainable_backbone_layers=config['trainable_backbone_layers'],
                                                      attention_per_attrib=config['attention_per_attrib'],
                                                      use_attention=config['use_attention'],
                                                      use_reduced_features_for_attrib=config['use_reduced_features_for_attrib'],
                                                      custom_box_predictor=config['custom_box_predictor'],
                                                      num_heads=config['num_heads'],
                                                      parallel_backbone=config['parallel_backbone'],
                                                      use_resnet101=config['use_resnet101_instead'])
    
    # load checkpoint weights
    checkpoint_file = f'checkpoints/{job_id}/epoch_{epoch_no}.pt'
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
  
    # test model
    c_frcnn_attn.evaluate(model, test_dataloader, device=device, calc_overall_metrics=True, print_cm=True, iou_threshold=0.5, score_threshold=0.7)

    # test on a single image for visual inspection
    if True:
        subfolder = os.path.join("eval_image", f"{job_id}")
        c_frcnn_attn.eval_image(raw_dataset, epoch_no, device, save=True, index=1083, target_attrib="category", 
                                show_attrib=True, model=model, subfolder=subfolder, eval_transform=get_transform(train=False),
                                score_threshold=0.1)
    

if __name__ == "__main__":

    print("running test_images.py")

    print(sys.argv)

    if len(sys.argv) >= 4:
        job_id = sys.argv[2]
        epoch_no = sys.argv[3]
        try:
            job_id = int(job_id)
            epoch_no = int(epoch_no)
        except:
            raise ValueError("Job ID and/or Epoch number are in an invalid format")
        test_images(job_id, epoch_no)

    else:
        test_images()