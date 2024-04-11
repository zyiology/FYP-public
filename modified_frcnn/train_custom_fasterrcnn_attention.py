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
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from test_images import test_images

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main():

    # annotations_filepath = 'data/mapped_gmaps_1_category_annotations.json'
    annotations_filepath = 'data/mapped_combined_annotations.json'
    print(f"loading annotations from '{annotations_filepath}'")
    num_classes = 2
    
    root = 'data/combined'
    logging = True
    save_checkpoints = True

    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    # TRAINING SCHEDULE
    # short
    # num_iterations = 16000 # number of images to process before stopping training
    # milestones = [9500,13000]

    # medium
    # num_iterations = 24000 # number of images to process before stopping training
    # milestones = [16000,20000]

    # medium long
    # num_iterations = 28000 # number of images to process before stopping training
    # milestones = [17000,23000]

    # long
    num_iterations = 32000 # number of images to process before stopping training
    milestones = [20000,26000]

    # exclude = list(range(1200,1782))
    exclude = None
    trainable_backbone_layers = 5 # range from 0 to 5 for resnet
    eval_image_attribute = None #"category"

    use_attention = True
    attention_per_attrib = True
    custom_box_predictor = True
    num_heads = 4

    parallel_backbone = True
    
    use_attribute_weights = True
    weight_scaling = (1/2)
    use_resnet101_instead = True
    use_pretrained_weights = False
    use_reduced_features_for_attrib = False # modification 1
    eval_attrib = True
    load_ids_from_file = True
    print(f"use attention to predict attributes: {use_attention}")
    print(f'self-attention layer for each attribute: {attention_per_attrib}')
    print(f"number of self-attention heads: {num_heads}")
    print(f"parallel backbone (mod4): {parallel_backbone}")
    print(f'use custom box predictor (i.e. attention before box prediction): {custom_box_predictor}')
    print(f"using attribute_weights: {use_attribute_weights}")
    print(f"using reduced box features for attribute prediction (mod1): {use_reduced_features_for_attrib}")
    print(f"use resnet101 instead of resnet50: {use_resnet101_instead}")
    print(f"trainable backbone layers: {trainable_backbone_layers}")
    print(f"using pretrained weights: {use_pretrained_weights}")
    print(f"Evaluating attribute predictions: {eval_attrib}")
    if exclude:
        print(f"excluding ids: {exclude[0]} to {exclude[-1]}")

    print(f"processing attributes: {','.join(attrib_mappings.keys())}")

    seed = 420
    generator = torch.Generator().manual_seed(seed)
    print(f"setting torch random seed to {seed}")

    # maybe i should raise error if no gpu
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        get_transform(train=False), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)
    dataset = CustomFRCNNAttentionDataset(
        root, 
        get_transform(train=True), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)
    
    if load_ids_from_file:
        val_id_file = 'data/val_ids.txt'
        with open(val_id_file, 'r') as f:
            val_ids = [int(i) for i in f.read().split(',')]
        val_dataset = torch.utils.data.Subset(raw_dataset, val_ids)

        train_id_file = 'data/train_ids.txt'
        with open(train_id_file, 'r') as f:
            train_ids = [int(i) for i in f.read().split(',')]
        train_dataset = torch.utils.data.Subset(raw_dataset, train_ids)
    else:
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    if use_attribute_weights:   
        attribute_weight_threshold = 3 # maybe set this as a parameter?

        # calculate attribute weights
        attribute_counts = {}
        for attrib_name, attrib_mapping in dataset.reverse_attrib_mappings.items():
            attribute_counts[attrib_name] = torch.zeros(len(attrib_mapping.keys()), device=device) # {k:0 for k in attrib_mapping.keys()}
        
        #for _, target in train_dataset:
        for index in train_dataset.indices:
            target = dataset.get_target(index)
            attribs = target['attributes']
            for attrib_name, attrib_vals in attribs.items():
                for val in attrib_vals:
                    # don't count 'unknown' attribute
                    if int(val) == 0:
                        continue

                    attribute_counts[attrib_name][int(val)] += 1

        print("attribute_counts", attribute_counts)

        # attributes_weights_dict has attribute names for keys and a tensor of weights for each value
        attribute_weights_dict = {}
        for attrib_name, counts in attribute_counts.items():
            # calculate log scale inverse frequency
            weights = torch.log(counts.sum() / counts + 1)

            # To account for infinite/very high weights, threshold is set
            # in practice, weights of the common classes are around 0.2-0.5
            # so don't set this too high
            weights[weights>attribute_weight_threshold] = 3
            weights = torch.nn.functional.normalize(weights, dim=0)
            
            print(f"weights scaled by {weight_scaling}")
            weights *= weight_scaling

            attribute_weights_dict[attrib_name] = weights
    else:
        attribute_weights_dict = None

    print("attribute_weights_dict", attribute_weights_dict)

    log_folder = os.path.join('logs', sys.argv[1])
    os.makedirs(log_folder)
    config = {
        "use_attention": use_attention,
        "attention_per_attrib": attention_per_attrib,
        "custom_box_predictor": custom_box_predictor,
        "num_heads": num_heads,
        "use_attribute_weights": use_attribute_weights,
        "use_resnet101_instead": use_resnet101_instead,
        "use_pretrained_weights": use_pretrained_weights,
        "use_reduced_features_for_attrib": use_reduced_features_for_attrib,
        "eval_attrib": eval_attrib,
        "trainable_backbone_layers": trainable_backbone_layers,
        "parallel_backbone": parallel_backbone
    }
    config_file = os.path.join(log_folder, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    if not load_ids_from_file:
        val_ids_log_file = os.path.join(log_folder, 'val_ids.txt')
        with open(val_ids_log_file, 'w') as f:
            val_ids = [str(t['image_id']) for _, t in val_dataset]
            f.write(",".join(val_ids))

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    validation_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

    if logging:
        runs_directory = os.path.join('runs', sys.argv[1])
        writer = SummaryWriter(runs_directory)

    if not os.path.isfile('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'):
        os.system('wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')

    if use_resnet101_instead:
        model = c_frcnn_attn.custom_fasterrcnn_resnet101_fpn(num_classes=num_classes,
                                                             attrib_mappings=attrib_mappings, 
                                                             attribute_weights_dict=attribute_weights_dict, 
                                                             trainable_backbone_layers=trainable_backbone_layers,
                                                             attention_per_attrib=attention_per_attrib,
                                                             pretrained=use_pretrained_weights,
                                                             use_attention=use_attention,
                                                             use_reduced_features_for_attrib=use_reduced_features_for_attrib,
                                                             custom_box_predictor=custom_box_predictor,
                                                             num_heads=num_heads,
                                                             parallel_backbone = parallel_backbone)
    else:
        model = c_frcnn_attn.custom_fasterrcnn_resnet50_fpn(num_classes=num_classes, 
                                                            attrib_mappings=attrib_mappings, 
                                                            attribute_weights_dict=attribute_weights_dict, 
                                                            trainable_backbone_layers=trainable_backbone_layers,
                                                            attention_per_attrib=attention_per_attrib,
                                                            pretrained=use_pretrained_weights,
                                                            use_attention=use_attention,
                                                            use_reduced_features_for_attrib=use_reduced_features_for_attrib,
                                                            custom_box_predictor=custom_box_predictor,
                                                            num_heads=num_heads,
                                                            parallel_backbone = parallel_backbone) # PARALLEL BACKBONE HANDLING NOT IMPLEMENTED

    if use_pretrained_weights:                                                        
        pretrained_dict = torch.load('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')  # Load the pretrained model weights
        model_dict = model.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        overwrite_eps(model, 0.0)

    model.to(device)

    num_epochs = math.ceil(num_iterations/len(train_dataset))
    print(f"training for {num_epochs} epochs...")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0003,
        # lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    milestones = [math.ceil(x/len(train_dataset)) for x in milestones]
    print("epoch milestones (where lr drops):", milestones)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1  # Factor to reduce the learning rate by at each milestone
    )

    subfolder = os.path.join("eval_image", f"{sys.argv[1]}")
    os.makedirs(subfolder)

    if save_checkpoints:
        checkpoints_folder = os.path.join("checkpoints", f"{sys.argv[1]}")
        os.makedirs(checkpoints_folder)

    n_attrib = len(attrib_mappings.keys())

    mAPs = np.zeros(num_epochs)
    f1s = np.zeros((num_epochs, n_attrib))
    overall_f1s = np.zeros((num_epochs, n_attrib))

    for epoch in range(num_epochs):
        # train the model for one epoch
        train_metrics = c_frcnn_attn.train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        coco_evaluator, f1 = c_frcnn_attn.evaluate(model, validation_dataloader, device=device, eval_attrib=eval_attrib, 
                                                   calc_overall_metrics=True, print_cm=True)

        mAPs[epoch] = coco_evaluator.coco_eval['bbox'].stats[0]
        f1s[epoch] = f1[:,0]
        overall_f1s[epoch] = f1[:,1]
        
        if logging:
            writer.add_scalar('Loss/train', train_metrics.loss.global_avg, epoch)
            writer.add_scalar('mAP/validation', coco_evaluator.coco_eval['bbox'].stats[0], epoch)
        
        # save image of example
        # eval_image(model, epoch, device)
        
        # c_frcnn_attn.eval_image(raw_dataset, epoch, device, save=True, index=689, target_attrib="roof_shape", 
        #             show_attrib=True, model=model, subfolder=subfolder, eval_transform=get_transform(train=False),
        #             score_threshold=0.6)

        if eval_image_attribute:

            c_frcnn_attn.eval_image(raw_dataset, epoch, device, save=True, index=729, target_attrib=eval_image_attribute, 
                        show_attrib=True, model=model, subfolder=subfolder, eval_transform=get_transform(train=False),
                        score_threshold=0.6)

        if save_checkpoints:

            # save model checkpoint
            checkpoint_file = os.path.join(checkpoints_folder, f"epoch_{epoch}.pt")
            
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_metrics.loss.global_avg
                        }, checkpoint_file)

    metrics_folder = 'metrics'
    metrics_file = os.path.join(metrics_folder, f"{sys.argv[1]}.png")

    x = np.arange(len(mAPs))
    plt.plot(x, mAPs, label='mAPs', marker='o')
    for i, attrib_name in enumerate(attrib_mappings.keys()):
        plt.plot(x, f1s[:,i], label=f'F1 Scores ({attrib_name})', marker='s')

    # plt.plot(x, f1s, label='F1 Scores', marker='s')
    
        plt.plot(x, overall_f1s[:,i], label=f'Overall F1 Scores ({attrib_name})', marker='^')

    plt.title('Performance Metrics Over Time')
    plt.xlabel('Time (index)')
    plt.ylabel('Value')
    plt.legend()  # This adds the legend to the plot

    # Save the plot to a file
    plt.savefig(metrics_file, dpi=300)  # Saves the plot as a PNG file
    plt.close()
    
    # if save_checkpoints:
    #     # keep only top 3 val_loss checkpoints
    #     best_epochs = np.argsort(-val_losses)[:3]
    #     files_to_keep = [f"epoch_{epoch}.pt" for epoch in best_epochs]
    #     for filename in os.listdir(checkpoints_folder):
    #         if filename not in files_to_keep:
    #             filepath = os.path.join(checkpoints_folder, filename)
    #             os.remove(filepath)


    if logging:
        writer.flush()
        writer.close()


if __name__ == "__main__":

    print("running train_custom_fasterrcnn_attention.py")

    server = True
    if server:
        main()
    else:
        with open('.output.log', 'w') as f:
            sys.stdout = f
            main()