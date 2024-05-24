### TRAINING SCRIPT FOR CUSTOM FASTER R-CNN WITH ATTENTION ###
# Modifications can be selected by toggling the flags at the start of the script
# Made for running on a slurm cluster - specifically, with jobID generated from running on the cluster
# jobID has to be provided as argument when calling this script otherwise

import json
import math
import sys
import os
import utils
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T
from torchvision.models.detection._utils import overwrite_eps

# from local files
import custom_faster_rcnn_attention as c_frcnn_attn
from custom_frcnn_dataset import CustomFRCNNAttentionDataset

# transforms that are applied to the images
def get_transform(train):
    transforms = []
    if train:
        # additional augmentations could be added
        transforms.append(T.RandomHorizontalFlip(0.5))
       
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main():
    '''
    Function that trains a modified Faster R-CNN model. Flags can be adjusted in the lines below this to
    modify the model. Model draws from images from  the `root` folder and annotations from `annotations_filepath`.
    Various files are saved during the training process, e.g. checkpoints, metrics, and a config file to recreate the model.
    '''
    
    # location of annotations file
    annotations_filepath = 'data/mapped_combined_annotations.json'
    print(f"loading annotations from '{annotations_filepath}'")

    # json mapping attribute classes to numbers for efficient encoding
    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)
    
    # dataset root directory - images should be stored in a subfolder called 'images'
    root = 'data/combined'

    # whether to log training metrics using tensorboard
    logging = True

    # whether to save model checkpoints (will be saved to subfolder 'data/checkpoints/[number of run]')
    save_checkpoints = True

    # TRAINING SCHEDULES - can be modified based on dataset
    # short
    # num_iterations = 16000 # number of images to process before stopping training
    # milestones = [9500,13000]

    # medium
    # num_iterations = 24000 # number of images to process before stopping training
    # milestones = [16000,20000]

    # long
    num_iterations = 32000 # number of images to process before stopping training
    milestones = [20000,26000] # number of images to process before stepping down the learning rate

    exclude = None # can be list of image ids to exclude from training (to speed up testinig)
    trainable_backbone_layers = 5 # range from 0 to 5 for resnet, lower values to lock more layers

    # during each validation phase, model is tested on a single image and the image is saved to disk
    # for visual inspection. because there may be multiple attributes that are predicted, 
    # this is the one that will be displayed on that saved image
    # can be set to None to not save any images
    eval_image_attribute = "category" 

    # number of classes in the dataset
    num_classes = 2

    # custom model flags
    use_attention = True # whether to use self-attention layer(s) between proposal tensor and attribute prediction
    attention_per_attrib = True # whether to use a separate self-attention layer for each attribute
    custom_box_predictor = True # whether to use attention before box prediction
    num_heads = 4 # number of self-attention heads
    parallel_backbone = True # whether to use a separate backbone for attribute prediction
    use_resnet101_instead = True # whether to use resnet101 instead of resnet50
    use_pretrained_weights = False # whether to use pretrained weights for the backbone/fast r-cnn
    eval_attrib = True # whether to evaluate attribute prediction performance
    load_ids_from_file = True # whether to load train/val/test set ids from file (vs random split, which doesn't generate test set)
    
    # whether to use reduced features for attribute prediction (mod1; as opposed to mod2 which uses full proposal tensor)
    use_reduced_features_for_attrib = False 
    
    # whether to calculate attribute weights for loss function (inverse frequency)
    # if False, will used fixed alpha for all classes (refer to focal loss paper for more info)
    use_attribute_weights = True 
    
    # scaling factor for attribute weights
    # should be scaled such that loss is proportional to classification/regression loss
    weight_scaling = (1/2)

    # print flags for logging
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

    # train on the GPU or on the CPU, if a GPU is not available
    # GPU is highly recommended for faster training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # unmodified dataset for evaluation
    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        get_transform(train=False), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)
    
    # dataset for training, uses different transform
    dataset = CustomFRCNNAttentionDataset(
        root, 
        get_transform(train=True), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)
    
    if load_ids_from_file:
        # load validation dataset
        val_id_file = 'data/val_ids.txt'
        with open(val_id_file, 'r') as f:
            val_ids = [int(i) for i in f.read().split(',')]
        val_dataset = torch.utils.data.Subset(raw_dataset, val_ids)

        # load train dataset
        train_id_file = 'data/train_ids.txt'
        with open(train_id_file, 'r') as f:
            train_ids = [int(i) for i in f.read().split(',')]
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
    else:
        # randomly split dataset into train and validation
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    if use_attribute_weights:   
        # set a maximum to prevent infinite/very high weights from distorting loss
        # in practice, weights of the common classes were observed to be around 0.2-0.5
        attribute_weight_threshold = 3

        # create empty dictionary to store attribute counts
        attribute_counts = {}
        for attrib_name, attrib_mapping in dataset.reverse_attrib_mappings.items():
            attribute_counts[attrib_name] = torch.zeros(len(attrib_mapping.keys()), device=device)
        
        # count attribute occurrences in training set
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

            weights[weights>attribute_weight_threshold] = 3

            # normalize weights to prevent attribute loss from growing with number of attributes
            # and potentially dominating the classification/regression loss
            weights = torch.nn.functional.normalize(weights, dim=0)
            
            print(f"weights scaled by {weight_scaling}")
            weights *= weight_scaling

            attribute_weights_dict[attrib_name] = weights
    else:
        attribute_weights_dict = None

    print("attribute_weights_dict", attribute_weights_dict)

    # save flags for this run into a file, so they can be used to reconstruct model for testing
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
        "parallel_backbone": parallel_backbone,
        "num_classes": num_classes
    }
    config_file = os.path.join(log_folder, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # save validation ids to file if random_split was used
    if not load_ids_from_file:
        val_ids_log_file = os.path.join(log_folder, 'val_ids.txt')
        with open(val_ids_log_file, 'w') as f:
            val_ids = [str(t['image_id']) for _, t in val_dataset]
            f.write(",".join(val_ids))

    # load data into dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    validation_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

    # if logging, create tensorboard writer
    if logging:
        runs_directory = os.path.join('runs', sys.argv[1])
        writer = SummaryWriter(runs_directory)

    # create model
    model = c_frcnn_attn.custom_fasterrcnn_resnet_fpn(num_classes=num_classes,
                                                      attrib_mappings=attrib_mappings, 
                                                      attribute_weights_dict=attribute_weights_dict, 
                                                      trainable_backbone_layers=trainable_backbone_layers,
                                                      attention_per_attrib=attention_per_attrib,
                                                      pretrained=use_pretrained_weights,
                                                      use_attention=use_attention,
                                                      use_reduced_features_for_attrib=use_reduced_features_for_attrib,
                                                      custom_box_predictor=custom_box_predictor,
                                                      num_heads=num_heads,
                                                      parallel_backbone=parallel_backbone,
                                                      use_resnet101=use_resnet101_instead)
    
    # load model to device
    model.to(device)

    num_epochs = math.ceil(num_iterations/len(train_dataset))
    print(f"training for {num_epochs} epochs...")

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0003,
        momentum=0.9,
        weight_decay=0.0005
    )

    milestones = [math.ceil(x/len(train_dataset)) for x in milestones]
    print("epoch milestones (where lr drops):", milestones)

    # construct a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1  # Factor to reduce the learning rate by at each milestone
    )

    # where to save the evaluation image (single image from validation set, for visual inspection)
    subfolder = os.path.join("eval_image", f"{sys.argv[1]}")
    os.makedirs(subfolder)

    # where to save checkpoints
    if save_checkpoints:
        checkpoints_folder = os.path.join("checkpoints", f"{sys.argv[1]}")
        os.makedirs(checkpoints_folder)

    # number of attributes being predicted
    n_attrib = len(attrib_mappings.keys())

    # arrays to store metrics over time, for plotting
    # could be moved to tensorboard instead, but server i was testing didn't support tensorboard
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

        # save metrics for plotting
        mAPs[epoch] = coco_evaluator.coco_eval['bbox'].stats[0]
        f1s[epoch] = f1[:,0]
        overall_f1s[epoch] = f1[:,1]
        
        # write metrics to tensorboard
        if logging:
            writer.add_scalar('Loss/train', train_metrics.loss.global_avg, epoch)
            writer.add_scalar('mAP/validation', coco_evaluator.coco_eval['bbox'].stats[0], epoch)
        
        # save image of example
        if eval_image_attribute:
            c_frcnn_attn.eval_image(raw_dataset, epoch, device, save=True, index=729, target_attrib=eval_image_attribute, 
                        show_attrib=True, model=model, subfolder=subfolder, score_threshold=0.6)

        if save_checkpoints:
            # save model checkpoint
            checkpoint_file = os.path.join(checkpoints_folder, f"epoch_{epoch}.pt")
            
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_metrics.loss.global_avg
                        }, checkpoint_file)


    # plot metrics
    metrics_folder = 'metrics'
    metrics_file = os.path.join(metrics_folder, f"{sys.argv[1]}.png")

    x = np.arange(len(mAPs))
    plt.plot(x, mAPs, label='mAPs', marker='o')
    for i, attrib_name in enumerate(attrib_mappings.keys()):
        plt.plot(x, f1s[:,i], label=f'F1 Scores ({attrib_name})', marker='s')    
        plt.plot(x, overall_f1s[:,i], label=f'Overall F1 Scores ({attrib_name})', marker='^')

    plt.title('Performance Metrics Over Time')
    plt.xlabel('Time (index)')
    plt.ylabel('Value')
    plt.legend()  # This adds the legend to the plot
    plt.savefig(metrics_file, dpi=300)  # Saves the plot as a PNG file
    plt.close()

    if logging:
        writer.flush()
        writer.close()


if __name__ == "__main__":

    print("running train_custom_fasterrcnn_attention.py")

    server = True # on slurm cluster, job output is automatically routed to file
    if server:
        main()
    else:
        # manually route output to file if not on slurm cluster
        # could be modified to increment jobIDs automatically and save to different files
        with open('.output.log', 'w') as f:
            sys.stdout = f
            main()