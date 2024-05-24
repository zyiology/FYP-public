## SCRIPT TO TRAIN BASELINE MODEL
# separate training and testing of detector and classifier
# modify flags to determine which should be trained

import math
import time
import sys
import utils
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import roi_align
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.resnet import resnet101, ResNet101_Weights
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F

# from local files
from engine import train_one_epoch, evaluate
from inception.inception_resnet_v2 import Inception_ResNetv2_multitask
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
from attrib_eval import print_confusion_matrix

def main():
    '''
    Function to train the separate components of the baseline model.
    For the detector, the model is trained on the bounding box annotations.
    For the classifier, the ground truth bounding boxes are used to crop the images,
    and the model is trained on the cropped images with the corresponding attribute labels.
    '''

    JOB_ID = sys.argv[1]

    annotations_filepath = 'data/mapped_combined_annotations.json'

    root = 'data/combined'

    # lr schedule
    num_iterations = 30000 # number of images to process before stopping training
    milestones = [16000,25000]

    # flags for training
    use_attribute_weights = True # calculate attribute weights for focal loss
    save_checkpoints = True # save model checkpoints
    use_predefined_trainval = True # load train and val ids from file (vs random_split)
    train_detector = True # whether to train the detector
    train_classifier = False # whether to train the classifier
    resnet101_detector = False # whether to use a pretrained detector
    trainable_backbone_layers = 5
    
    print(f"train_detector: {train_detector}") # uses pretrained weights, unless resnet101 is used
    print(f"resnet101 detector: {resnet101_detector}")
    print(f"train_classifier: {train_classifier}")
    print(f"use_attribute_weights={use_attribute_weights}")
    print(f"trainable_backbone_layers={trainable_backbone_layers}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed = 420
    generator = torch.Generator().manual_seed(seed)
    print(f"setting torch random seed to {seed}")

    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    exclude = None # list of ids to exclude from training (to speed up training)
    if exclude:
        assert not use_predefined_trainval, "cannot exclude ids if using predefined train and val set"
        print(f"excluding ids from {exclude[0]} to {exclude[-1]}")

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
        
    dataset_size = len(dataset)

    # load train and val ids from file if using predefined train and val set - highly recommended
    if use_predefined_trainval:
        train_id_file = 'data/train_ids.txt'
        val_id_file = 'data/val_ids.txt'

        with open(train_id_file, 'r') as f:
            train_ids = [int(i) for i in f.read().split(',')]

        with open(val_id_file, 'r') as f:
            val_ids = [int(i) for i in f.read().split(',')]
        
        val_dataset = torch.utils.data.Subset(raw_dataset, val_ids)
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
    
    else:
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # calculate number of epochs to reduce lr/end training
    num_epochs = math.ceil(num_iterations/len(train_dataset))
    milestones = [math.ceil(x/len(train_dataset)) for x in milestones]

    # load data into dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)
    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # same as in train_custom_fasterrcnn_attention, refer there
    if use_attribute_weights:   
        attribute_weight_threshold = 3

        # calculate attribute weights
        attribute_counts = {}
        for attrib_name, attrib_mapping in dataset.reverse_attrib_mappings.items():
            attribute_counts[attrib_name] = torch.zeros(len(attrib_mapping.keys()), device=device)

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

            weights[weights>attribute_weight_threshold] = attribute_weight_threshold
            weights = torch.nn.functional.normalize(weights, dim=0)

            attribute_weights_dict[attrib_name] = weights
        
        print("attribute_weights_dict", attribute_weights_dict)
    else:
        attribute_weights_dict = None

    num_classes_dict = {k:len(v) for k,v in attribute_weights_dict.items()}        

    # TRAIN DETECTOR
    if train_detector:
        if save_checkpoints:
            detector_checkpoints_folder = os.path.join("checkpoints", f"{sys.argv[1]}", "detector")
            os.makedirs(detector_checkpoints_folder)

        if resnet101_detector:
            # replace backbone with resnet101
            detector = fasterrcnn_resnet50_fpn()
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=True,
                                 norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            detector.backbone = backbone
        else:
            detector = fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=trainable_backbone_layers)

        detector.roi_heads.box_predictor = FastRCNNPredictor(in_channels=1024,num_classes=2)

        detector.to(device)

        if resnet101_detector:
            # construct an optimizer
            params = [p for p in detector.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=0.0003, # for resnet101
                momentum=0.9,
                weight_decay=0.0005
            )

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=0.1  # Factor to reduce the learning rate by at each milestone
            )
            print(f"training detector for {num_epochs} epochs...")
            print(f"learning rate stepped down at epochs {milestones}")
        else:
            # because using pretrained detector, don't need as much training
            params = [p for p in detector.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=0.001, # for resnet50
                momentum=0.9,
                weight_decay=0.0005
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR( # steps down every 5 epochs
               optimizer,
               step_size=5,
               gamma=0.1
            )
            num_epochs = 14
            print(f"training detector for {num_epochs} epochs...")
        
        # store mAPs for each epoch for plotting
        mAPs = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(detector, optimizer, train_dataloader, device, epoch, print_freq=100)
            lr_scheduler.step()
            coco_evaluator = evaluate(detector, validation_dataloader, device=device)
            mAPs[epoch] = coco_evaluator.coco_eval['bbox'].stats[0]
            if save_checkpoints:

                # save model checkpoint
                checkpoint_file = os.path.join(detector_checkpoints_folder, f"epoch_{epoch}.pt")
                
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': detector.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_metrics.loss.global_avg
                            }, checkpoint_file)
        
        # plot mAPs to an image
        metrics_folder = 'metrics'
        metrics_file = os.path.join(metrics_folder, f"{sys.argv[1]}.png")
        x = np.arange(len(mAPs))
        plt.plot(x, mAPs, label='mAPs', marker='o')
        plt.title('Performance Metrics Over Time')
        plt.xlabel('Time (index)')
        plt.ylabel('Value')
        plt.legend()  # This adds the legend to the plot

        # Save the plot to a file
        plt.savefig(metrics_file, dpi=300)  # Saves the plot as a PNG file
        plt.close()

    # TRAIN CLASSIFIER
    if train_classifier:
        if save_checkpoints:
            classifier_checkpoints_folder = os.path.join("checkpoints", f"{sys.argv[1]}", "classifier")
            os.makedirs(classifier_checkpoints_folder)

        print(f"training classifier for {num_epochs} epochs...")
        print(f"learning rate steps down at epochs {milestones}")
        
        classifier = Inception_ResNetv2_multitask(in_channels=3, classes=num_classes_dict)
        classifier.to(device)

        # construct an optimizer
        params = [p for p in classifier.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.001, # for inception resnet v2
            momentum=0.9,
            weight_decay=0.0005
        )
	
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
             optimizer,
             milestones=milestones,
             gamma=0.1  # Factor to reduce the learning rate by at each milestone
        )

        # store metrics for plotting
        num_attribs = len(attrib_mappings)
        f1s = np.zeros((num_epochs, num_attribs))
        x = np.arange(num_epochs)

        for epoch in range(num_epochs):
            train_metrics = classifier_train_one_epoch(classifier, optimizer, train_dataloader, device, epoch, print_freq=100, attribute_weights_dict=attribute_weights_dict)
            lr_scheduler.step()
            f1 = classifier_evaluate(classifier, validation_dataloader, device=device, attribute_mappings=attrib_mappings)
            f1s[epoch] = f1

            if save_checkpoints:

                # save model checkpoint
                checkpoint_file = os.path.join(classifier_checkpoints_folder, f"epoch_{epoch}.pt")
                
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_metrics.loss.global_avg
                            }, checkpoint_file)

        # plot evaluation metrics for each epoch
        metrics_folder = 'metrics'
        metrics_file = os.path.join(metrics_folder, f"{sys.argv[1]}.png")
        for i, attrib_name in enumerate(attrib_mappings.keys()):
            plt.plot(x, f1s[:,i], label=f'F1 Scores ({attrib_name})', marker='s')

        plt.title('Performance Metrics Over Time')
        plt.xlabel('Time (index)')
        plt.ylabel('Value')
        plt.legend()  # This adds the legend to the plot

        # Save the plot to a file
        plt.savefig(metrics_file, dpi=300)  # Saves the plot as a PNG file
        plt.close()

# adapted from train_one_epoch from engine.py in torchvision
def classifier_train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, attribute_weights_dict, scaler=None, crop_size=(299,299)):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        
        target_attrib_dict = defaultdict(list)
        images_cropped = []
        ids = []
        for i, t in zip(images,targets):
            ids.append(t['image_id'])
            indices = []

            # IMPT: code below assumes there is only one labeled box per image, and that
            # box has all given attributes labelled - should be possible to adapt to multiple boxes
            # but done this way to equalise the "weightage" of each image

            # for the given image/target, get the attribute data for the one labeled box
            # check if all attributes are known
            # if any are unknown, skip this image, don't use it for training
            skip = False
            for target_attrib in attribute_weights_dict.keys():
                attrib = t['attributes'][target_attrib]

                attrib = attrib.to(device)

                # there should only be one non-zero value in attrib
                index = torch.argmax(attrib)

                if attrib[index] == 0:
                    # annotation for this attribute is unknown, cannot be used for training
                    skip = target_attrib
                    break
                    
                indices.append(index)
                target_attrib_dict[target_attrib].append(attrib[index])

            if skip is not False:
                # undo added attribs in this loop
                for target_attrib in attribute_weights_dict.keys():
                    if skip == target_attrib: break
                    target_attrib_dict[target_attrib].pop()

                # skip this image
                continue

            assert all(torch.equal(indices[0], tensor) for tensor in indices[1:]), "not all indices match! something wrong with retrieving attribute and corresponding bbox"
            # crop image
            # x,y,w,h = t['boxes'][index]
            box = [t['boxes'][index].unsqueeze(0).to(device)]
            
            image = i.unsqueeze(0)

            cropped_image = roi_align(image, box, output_size=crop_size)
            images_cropped.append(cropped_image.squeeze())

        for k,v in target_attrib_dict.items():
            target_attrib_dict[k] = torch.stack(v)

        images = torch.stack(images_cropped)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # send images through model
            classes_raw_dict = model(images)
            
            # apply softmax to classes
            classes_logits_dict = {k:F.softmax(v, dim=1) for k,v in classes_raw_dict.items()}

            alpha = 0.25
            gamma = 2
            total_loss = torch.tensor(0., dtype=torch.float32).to(device)

            for key in attribute_weights_dict:
                # override the default alpha with the attribute_weights
                alpha = attribute_weights_dict[key]

                # how about i just remove the unknowns at this step instead of skipping image like earlier?
                target_attrib = target_attrib_dict[key]
                logits_attrib = classes_logits_dict[key]

                # Skip if attrib_logits and attrib are empty
                if target_attrib.numel() == 0 and logits_attrib.numel() == 0:
                    continue

                ## calculate focal loss between logits and target

                ce_loss = F.cross_entropy(logits_attrib, target_attrib, reduction='none')
                if math.isnan(ce_loss): # skip this attribute if loss is nan
                    continue

                pt = torch.exp(-ce_loss)

                if (type(alpha) == torch.Tensor and len(alpha)>1):
                    # Index alpha with the class labels to get a tensor of size N
                    alpha_selected = alpha[target_attrib]
                    
                    # Ensure focal_factor is broadcastable with ce_loss
                    # ce_loss is of size N, so we squeeze focal_factor to match this shape
                    focal_factor = (1 - pt) ** gamma
                    # focal_factor = focal_factor.squeeze()

                    # Compute the focal loss
                    focal_loss = alpha_selected * focal_factor * ce_loss

                else:
                    focal_loss = alpha * (1 - pt) ** gamma * ce_loss


                loss = focal_loss.mean()

                if not math.isfinite(loss.item()):
                    print(f"Loss is {loss.item()}, stopping training")
                    sys.exit(1)
        
                total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"]) # maybe need to make loss some kind of dict to have it report each attrib loss separately

    return metric_logger

    # some unadapted code from original function
    # the stuff dealing with multi-GPU training was removed for ease of use
    # 
    #     # reduce losses over all GPUs for logging purposes
    #     loss_dict_reduced = utils.reduce_dict(loss_dict)
    #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    #     loss_value = losses_reduced.item()

    #     if not math.isfinite(loss_value):
    #         print(f"Loss is {loss_value}, stopping training")
    #         print(loss_dict_reduced)
    #         sys.exit(1)

    #     optimizer.zero_grad()
    #     if scaler is not None:
    #         scaler.scale(losses).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #     else:
    #         losses.backward()
    #         optimizer.step()

    #     if lr_scheduler is not None:
    #         lr_scheduler.step()

    #     metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # return metric_logger

# based on evaluate from engine.py in torchvision
@torch.inference_mode()
def classifier_evaluate(model, data_loader, device, attribute_mappings, crop_size=(299,299)):
    # crop_size refers to size of cropped and resized building

    # n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    gt = defaultdict(list)
    dt = defaultdict(list)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        
        target_attrib_dict = defaultdict(list)
        images_cropped = []
        
        for i, t in zip(images,targets):

            # IF DATASET HAS MULTIPLE BOXES WITH GT ATTRIBUTES (excluding unknowns)
            # need to modify this to crop out multiple gts
            # CURRENTLY ONLY CROPS OUT ONE GT PER IMAGE
            for target_attrib in attribute_mappings.keys():
                attrib = t['attributes'][target_attrib]

                attrib = attrib.to(device)

                # there should only be one non-zero value in attrib
                index = torch.argmax(attrib)
                target_attrib_dict[target_attrib].append(attrib[index])

            # crop image
            # x,y,w,h = t['boxes'][index]
            # print("t['boxes'][index]", t['boxes'][index])
            box = [t['boxes'][index].unsqueeze(0).to(device)]
            
            image = i.unsqueeze(0)

            cropped_image = roi_align(image, box, output_size=crop_size)
            images_cropped.append(cropped_image.squeeze())

        targets_dict = {k:torch.stack(v).to(cpu_device) for k,v in target_attrib_dict.items()}
        images = torch.stack(images_cropped)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs_raw_dict = model(images)

        outputs_logits_dict = {k:F.softmax(v, dim=1) for k,v in outputs_raw_dict.items()}

        # get predicted class
        predicted_dict = {k:torch.argmax(v, dim=1).to(cpu_device) for k,v in outputs_logits_dict.items()}

        model_time = time.time() - model_time


        assert predicted_dict.keys() == targets_dict.keys(), "keys from output of model doesn't match keys in targets"
        
        for attrib_name in predicted_dict.keys():
            gt[attrib_name].append(targets_dict[attrib_name])
            dt[attrib_name].append(predicted_dict[attrib_name])

        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    f1s = np.zeros(len(gt.keys()))

    for attrib_index, attrib_name in enumerate(gt.keys()):
        print(f"evaluating {attrib_name}")
        predicted = torch.cat(dt[attrib_name])
        targets = torch.cat(gt[attrib_name])
        n = len(attribute_mappings[attrib_name])
        precision = torch.zeros(n, dtype=torch.float32)
        recall = torch.zeros(n, dtype=torch.float32)
        f1 = torch.zeros(n, dtype=torch.float32)
        counts = torch.zeros(n, dtype=torch.int32)

        for i in range(n):
            counts[i] = (targets == i).sum().item()

            # Calculate TP, FP, FN for the class of interest
            TP = ((predicted == i) & (targets == i)).sum().item()
            FP = ((predicted == i) & (targets != i)).sum().item()
            FN = ((predicted != i) & (targets == i)).sum().item()

            if TP+FP==0:
                precision[i]=0
            else:
                precision[i] = TP/(TP+FP)

            if TP+FN==0:
                recall[i]=-1
            else:
                recall[i] = TP / (TP+FN)

            if precision[i]==-1 or recall[i]==-1 or precision[i]+recall[i]==0:
                f1[i] = 0
            else:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

            print(f"for class {i}: precision={precision[i]:0.3f}, recall={recall[i]:0.3f}, f1={f1[i]:0.3f}, count={counts[i]}")

        weights = counts / counts.sum()

        # calculate weighted averages

        wavg_precision = (weights * precision).sum().item()
        wavg_recall = (weights * recall).sum().item()
        wavg_f1 = (weights * f1).sum().item()
        print(f"weighted averages: precision={wavg_precision:0.3f}, recall={wavg_recall:0.3f}, f1={wavg_f1:0.3f}")
        f1s[attrib_index] = wavg_f1

        # calculate confusion matrix given ground truth and predicted labels
        confusion_matrix = torch.zeros(n, n, dtype=torch.int32)
        for i, j in zip(targets, predicted):
            confusion_matrix[i, j] += 1

        # print the confusion matrix
        print_confusion_matrix(confusion_matrix, class_labels=None)

    return f1s

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)    


if __name__ == "__main__":
    main()
