from inception.inception_resnet_v2 import Inception_ResNetv2_multitask
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate
import sys
import utils
import torch
import json
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import roi_align
import math
import time
from collections import defaultdict
import os
from attrib_eval import AttribEvaluator, print_confusion_matrix
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.resnet import resnet101, ResNet101_Weights
from torchvision.ops import misc as misc_nn_ops
import numpy as np
import matplotlib.pyplot as plt

def main():

    JOB_ID = sys.argv[1]

    annotations_filepath = 'data/mapped_combined_annotations.json'
    #annotations_filepath = 'data/gmaps_1_annotations.json'
    root = 'data/combined'
    logging = True
    # num_epochs = 12

    # lr schedule
    num_iterations = 30000 # number of images to process before stopping training
    milestones = [16000,25000]

    use_attribute_weights = True
    save_checkpoints = True
    use_predefined_trainval = True
    train_detector = True
    train_classifier = False
    pretrained_detector = False
    
    print(f"train_detector: {train_detector}")
    print(f"pretrained detector: {pretrained_detector}")

    print(f"train_classifier: {train_classifier}")

    print(f"use_attribute_weights={use_attribute_weights}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed = 420
    generator = torch.Generator().manual_seed(seed)
    print(f"setting torch random seed to {seed}")

    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    # target_attrib = 'category'
    # num_classes = len(attrib_mappings[target_attrib])

    # exclude = list(range(700,2000))
    exclude = None

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

    if use_predefined_trainval:
        train_id_file = 'data/train_ids.txt'
        val_id_file = 'data/val_ids.txt'

        with open(train_id_file, 'r') as f:
            train_ids = [int(i) for i in f.read().split(',')]

        with open(val_id_file, 'r') as f:
            val_ids = [int(i) for i in f.read().split(',')]
        
        val_dataset = torch.utils.data.Subset(dataset, val_ids)
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
    
    else:
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    num_epochs = math.ceil(num_iterations/len(train_dataset))
    milestones = [math.ceil(x/len(train_dataset)) for x in milestones]
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)
    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

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

            # weights = counts.sum() / counts
            weights[weights>attribute_weight_threshold] = attribute_weight_threshold
            weights = torch.nn.functional.normalize(weights, dim=0)
            
            # print("doubling weights")
            # weights *= 2

            attribute_weights_dict[attrib_name] = weights
        
        print("attribute_weights_dict", attribute_weights_dict)
    else:
        attribute_weights_dict = None

    if logging:
        writer = SummaryWriter(f'runs/{JOB_ID}')

    num_classes_dict = {k:len(v) for k,v in attribute_weights_dict.items()}        

    if train_detector:
        if save_checkpoints:
            detector_checkpoints_folder = os.path.join("checkpoints", f"{sys.argv[1]}", "detector")
            os.makedirs(detector_checkpoints_folder)

        if pretrained_detector:
            print("USING RESNET-50, no pretrained faster r-cnn on resnet101")
            detector = fasterrcnn_resnet50_fpn(weights="DEFAULT")
            # backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=progress,
            #                      norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        else:
            detector = fasterrcnn_resnet50_fpn()
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=True,
                                 norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            trainable_backbone_layers=5
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            detector.backbone = backbone

        detector.roi_heads.box_predictor = FastRCNNPredictor(in_channels=1024,num_classes=2)

        detector.to(device)

        if not pretrained_detector:
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
            params = [p for p in detector.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=0.001, # for resnet101
                momentum=0.9,
                weight_decay=0.0005
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
               optimizer,
               step_size=5,
               gamma=0.1
            )
            num_epochs = 14
            print(f"training detector for {num_epochs} epochs...")
        

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

        #lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer,
        #    step_size=3,
        #    gamma=0.1
        #)
	
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
             optimizer,
             milestones=milestones,
             gamma=0.1  # Factor to reduce the learning rate by at each milestone
        )

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
        

        # targets = [t['attributes'][target_attrib] for t in targets]
        target_attrib_dict = defaultdict(list)
        images_cropped = []
        ids = []
        for i, t in zip(images,targets):
            ids.append(t['image_id'])
            indices = []

            skip = False

            for target_attrib in attribute_weights_dict.keys():
                attrib = t['attributes'][target_attrib]

                attrib = attrib.to(device)

                # there should only be one non-zero value in attrib
                index = torch.argmax(attrib)

                # assert attrib[index]!=0, f"attrib is 0 for image_{t['image_id']}, attrib looks like {attrib}"

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

                # print(target_attrib_dict)
                # print(images_cropped)

                # skip this image
                continue

            assert all(torch.equal(indices[0], tensor) for tensor in indices[1:]), "not all indices match! something wrong with retrieving attribute and corresponding bbox"
            # crop image
            # x,y,w,h = t['boxes'][index]
            # print("t['boxes'][index]", t['boxes'][index])
            box = [t['boxes'][index].unsqueeze(0).to(device)]
            
            image = i.unsqueeze(0)

            cropped_image = roi_align(image, box, output_size=crop_size)
            images_cropped.append(cropped_image.squeeze())

        for k,v in target_attrib_dict.items():
            target_attrib_dict[k] = torch.stack(v)

        # print('images len:', len(images_cropped))
        # print('attr len:', len(target_attrib_dict['category']))
        
        # targets = torch.stack(target_attrib_dict)
        images = torch.stack(images_cropped)
        # print('images shape:', images.shape)
        # print(images.device)
        # print(targets.device)
        # print(target_attrib_dict[0].device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # for image in images_cropped:
            #     pred = model(image)
            classes_raw_dict = model(images)
            
            # apply softmax to classes
            classes_logits_dict = {k:F.softmax(v, dim=1) for k,v in classes_raw_dict.items()}

            # print('classes', classes)
            # print('targets', targets)
            # print('ids', ids)

            # calculate loss
            # loss = F.cross_entropy(classes, targets)
            # losses = sum(loss)

            alpha = 0.25
            gamma = 2
            total_loss = torch.tensor(0., dtype=torch.float32).to(device)

            for key in attribute_weights_dict:
                # if attribute_weights_dict is defined, override the default alpha
                # if attribute_weights_dict:
                alpha = attribute_weights_dict[key]

                # attrib_logits = attribute_logits_dict[key]
                # attrib = attributes[key]

                target_attrib = target_attrib_dict[key]
                logits_attrib = classes_logits_dict[key]
                

                # Skip if attrib_logits and attrib are empty
                if target_attrib.numel() == 0 and logits_attrib.numel() == 0:
                    continue

                ce_loss = F.cross_entropy(logits_attrib, target_attrib, reduction='none')

                if math.isnan(ce_loss):
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

                    # print("alpha_selected", alpha_selected)
                    # print("(1 - pt)", (1 - pt))
                    # print("focal_factor", focal_factor)
                    # print("ce_loss", ce_loss)
                    # exit()

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

@torch.inference_mode()
def classifier_evaluate(model, data_loader, device, attribute_mappings, crop_size = (299,299)):
    n_threads = torch.get_num_threads()

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

        # for handling multiple gts in image
        # common_indices = None
        #     for target_attrib in attribute_mappings.keys():
        #         attrib = t['attributes'][target_attrib]

        #         attrib = attrib.to(device)

        #         # there should only be one non-zero value in attrib
        #         # index = torch.argmax(attrib)
        #         indices = set(torch.nonzero(attrib))
        #         if common_indices:
        #             common_indices = common_indices.intersection(indices)
        #         else:
        #             common_indices = indices

        #     for index in common_indices:
        #         target_attrib_dict[target_attrib].append(attrib[index])

        targets_dict = {k:torch.stack(v).to(cpu_device) for k,v in target_attrib_dict.items()}
        images = torch.stack(images_cropped)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs_raw_dict = model(images)

        outputs_logits_dict = {k:F.softmax(v, dim=1) for k,v in outputs_raw_dict.items()}

        # get predicted class
        predicted_dict = {k:torch.argmax(v, dim=1).to(cpu_device) for k,v in outputs_logits_dict.items()}

        # [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # evaluator_time = time.time()

        assert predicted_dict.keys() == targets_dict.keys(), "keys from output of model doesn't match keys in targets"
        
        for attrib_name in predicted_dict.keys():
            gt[attrib_name].append(targets_dict[attrib_name])
            dt[attrib_name].append(predicted_dict[attrib_name])

        # evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    f1s = np.zeros(len(gt.keys()))
    # overall_f1s = np.zeros(len(gt.keys()))

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
            # mask = torch.eq(targets, i)
            # predicted_masked = predicted[mask]

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



    # # calculate precision, recall, f1 for predicted and targets
    # tp = torch.sum(predicted == targets).item()
    # fp = torch.sum(predicted != targets).item()
    # fn = torch.sum(predicted != targets).item()

    # if tp+fp == 0:
    #     precision = -1
    # else:
    #     precision = tp / (tp + fp)

    # if tp+fn == 0:
    #     recall = -1
    # else:
    #     recall = tp / (tp + fn)

    # if precision+recall == 0:
    #     f1 = -1
    # else:
    #     f1 = 2 * (precision * recall) / (precision + recall)

    # print(f"precision: {precision}, recall: {recall}, f1: {f1}\n")

    # # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
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
