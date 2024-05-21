## SCRIPT TO TEST BASELINE MODEL
# detector and classifier models trained in baseline_custom_train.py
# models will be loaded and tested in this script
# uses batch size of 1 for evaluation

import os
import json
import utils
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.resnet import resnet101
from torchvision.ops import misc as misc_nn_ops

# from local files
from attrib_eval import print_confusion_matrix, compute_iou
from baseline_custom_train import classifier_evaluate
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
from engine import evaluate
from inception.inception_resnet_v2 import Inception_ResNetv2_multitask

@torch.inference_mode()
def main():
    # uses job ids and epochs to find corresponding checkpoint file
    detector_job_id = 215575
    detector_epoch = 24
    detector_resnet101 = True # whether detector uses resnet101 or resnet50 for backbone
    classifier_job_id = 215628
    classifier_epoch = 13
    print(f"Using detector from {detector_job_id}, epoch={detector_epoch}")
    print(f"Using classifier from {classifier_job_id}, epoch={classifier_epoch}")
    detector_checkpoint = os.path.join("checkpoints", f"{detector_job_id}", "detector", f"epoch_{detector_epoch}.pt")
    classifier_checkpoint = os.path.join("checkpoints", f"{classifier_job_id}", "classifier", f"epoch_{classifier_epoch}.pt")

    annotations_filepath = 'data/mapped_combined_annotations.json'
    root = 'data/combined'
    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)
    exclude = None

    # count the number of classes per attribute
    num_classes_dict = {}
    for k,v in attrib_mappings.items():
        num_classes_dict[k] = len(v)

    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        get_transform(train=False), 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=exclude)

    # load test dataset from disk
    test_id_file = 'data/test_ids.txt'
    with open(test_id_file, 'r') as f:
        test_ids = [int(i) for i in f.read().split(',')]
    test_dataset = torch.utils.data.Subset(raw_dataset, test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")

    print("======EVALUATE OBJECT DETECTION======")

    detector = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if detector_resnet101: # if model was trained with resnet101 backbone, load a resnet101 backbone
        backbone = resnet101(progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(backbone, 5)
        detector.backbone = backbone
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_channels=1024,num_classes=2)
    checkpoint = torch.load(detector_checkpoint)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.to(device)

    # evaluate detection performance
    evaluate(detector, test_dataloader, device=device)

    print("======EVALUATE ATTRIBUTE PREDICTION=======")

    classifier = Inception_ResNetv2_multitask(in_channels=3, classes=num_classes_dict)
    checkpoint = torch.load(classifier_checkpoint)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)

    # evaluate classification performance
    classifier_evaluate(classifier, test_dataloader, device, attrib_mappings)

    print("=======EVALUATE OVERALL PERFORMANCE=======")

    model = DetectorAndClassifier(detector, classifier)
    model.to(device)
    model.eval()

    gt = defaultdict(list)
    dt = defaultdict(list)
    total_false_pos = 0
    total_false_neg = 0
    total_num_gt = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for images, targets in metric_logger.log_every(test_dataloader, 25, header):
        images = list(img.to(device) for img in images)

        model_time = time.time()

        # process the images in the current batch and store detection performance statistics
        target_attribs, pred_attribs, false_pos, false_neg, num_gt = model(images, targets)
        total_false_pos += false_pos
        total_false_neg += false_neg
        total_num_gt += num_gt

        # store the ground truth and predicted attributes
        for k,v in pred_attribs.items():
            pred_attribs[k] = v.to(cpu_device)

        for k in target_attribs.keys():
            dt[k].append(pred_attribs[k].to(cpu_device))
            gt[k].append(target_attribs[k])
            
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)

    for k,v in dt.items():
        dt[k] = torch.cat(v)
    
    for k,v in gt.items():
        gt[k] = torch.cat(v)

    for k in gt.keys():
        assert len(gt[k]) == len(dt[k]), "length of predicted attributes don't match ground truth attributes"

    false_pos_per_image = total_false_pos/len(test_dataset)
    false_neg_per_image = total_false_neg/len(test_dataset)
    num_gts_per_image = total_num_gt/len(test_dataset)
    print(f"unmatched predictions per image: {false_pos_per_image}")
    print(f"unmatched gts per image: {false_neg_per_image}")
    print(f"Mean labels per image: {num_gts_per_image}")

    # evaluate attribute prediction performance for each attribute
    for attrib_name in gt.keys():
        print(f"evaluating {attrib_name}")
        predicted = dt[attrib_name]
        targets = gt[attrib_name]
        n = len(attrib_mappings[attrib_name])
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

        # calculate confusion matrix given ground truth and predicted labels
        confusion_matrix = torch.zeros(n, n, dtype=torch.int32)
        for i, j in zip(targets, predicted):
            confusion_matrix[i, j] += 1

        # print the confusion matrix
        print_confusion_matrix(confusion_matrix, class_labels=None)


# combined model of detector and classifier
# passes image through detector, then crops out predicted boxes with score>conf_threshold
# predicted boxes are matched to ground truth boxes, and false positives/negatives are identified
# successful matches are sent to classifier to be classified, then compared to ground truth attribute
class DetectorAndClassifier(nn.Module):
    def __init__(self, detector, classifier, conf_threshold=0.7, iou_threshold=0.5):
        super(DetectorAndClassifier, self).__init__()
        self.detector = detector
        self.classifier = classifier
        # Assume the classifier is designed to take fixed-size inputs,
        # e.g., the size of the crop from the bounding boxes.
        self.crop_size = (299, 299)  # Example crop size for the classifier
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def forward(self, images, targets):
        # images: Tensor[N, C, H, W]
        # targets: List[Dict]
        # process the images with the detector to find the bounding boxes
        # crop the images based on the bounding boxes
        # classify the cropped images with the classifier

        # Put the model in eval mode for inference to disable dropout, etc.
        self.detector.eval()
        with torch.no_grad():
            detections = self.detector(images)

        # example of detections[0]
        # {'boxes': tensor([[ 605.4993,   37.1151, 1192.0000,  384.9551], [ 651.5644,    0.0000,  950.7262,  576.1069],
        #             ... [ 507.8282,  425.5512, 1192.0000,  730.0000], [   0.0000,    0.0000,  427.4406,  398.8574]]), 
        #  'labels': tensor([1, 1, ..., 1, 1]), 
        #  'scores': tensor([0.3550, 0.3504, ..., 0.2879, 0.2866])}

        cropped_images = []
        target_attribs = defaultdict(list)
        false_pos = 0
        false_neg = 0
        num_gts = 0

        # process the detections to get the bounding boxes
        assert len(detections) == len(targets), "length of detections != targets??"
        for d, target, image in zip(detections, targets, images):
            # pick out boxes with confidence score > 0.7?
            mask = d['scores']>self.conf_threshold
            boxes = d['boxes'][mask]
            scores = d['scores'][mask]

            # sort scores and boxes by scores
            s = torch.argsort(scores)
            boxes = boxes[s]
            scores = scores[s]

            # logic adapted from cocoeval.py in pycocotools
            # match each pred box with a gt box
            pred_gt_box_mapping = {}
            
            for i, pred_box in enumerate(boxes):
                max_match_iou = -1
                for j, gt_box in enumerate(target['boxes']):
                    if j in pred_gt_box_mapping.values():
                        continue
                    iou = compute_iou(gt_box, pred_box)
                    if iou>self.iou_threshold and iou>max_match_iou:
                        pred_gt_box_mapping[i] = j
                        max_match_iou = iou
                if i not in pred_gt_box_mapping.keys():
                    false_pos += 1
                else:
                    skip = False
                    # check if gt box has attribute data
                    for k,v in target['attributes'].items():
                        if v[pred_gt_box_mapping[i]] == 0:
                            skip = True

                    if not skip:
                        box = [pred_box.unsqueeze(0)]
                        cropped_image = roi_align(image.unsqueeze(0), box, output_size=self.crop_size)
                        cropped_images.append(cropped_image)
                        for k,v in target['attributes'].items():
                            target_attribs[k].append(v[pred_gt_box_mapping[i]]) 

            assert len(target['boxes']) - len(pred_gt_box_mapping) >= 0, "some gt box(es) mapped to multiple predicted boxes"
            false_neg += len(target['boxes']) - len(pred_gt_box_mapping)
            num_gts += len(target['boxes'])


        for k,v in target_attribs.items():
            target_attribs[k] = torch.stack(v)
         
        # classify the cropped images with the classifier
        # Put the model in eval mode for inference to disable dropout, etc.
        self.classifier.eval()

        # process the cropped images with the classifier
        pred_attribs = defaultdict(list)
        for cropped_image in cropped_images:
            output_raw_dict = self.classifier(cropped_image)
            output_logits_dict = {k:F.softmax(v, dim=1) for k,v in output_raw_dict.items()}

            attrib_dict = {k:torch.argmax(v, dim=1) for k,v in output_logits_dict.items()}

            for k,v in attrib_dict.items():
                pred_attribs[k].append(v.squeeze())

        for k,v in pred_attribs.items():
            pred_attribs[k] = torch.stack(v)
            if len(pred_attribs[k].shape) != 1:
                print(pred_attribs[k])
                raise ValueError("not squeezed")


        return target_attribs, pred_attribs, false_pos, false_neg, num_gts
    
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)    


if __name__ == "__main__":
    print("running baseline_custom.py...")
    main()
