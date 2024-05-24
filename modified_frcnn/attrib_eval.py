## SCRIPT FOR EVALUATION OF ATTRIBUTE/OVERALL PREDICTION PERFORMANCE
# class to evaluate attribute classification for a given attribute

import torch
import math
from typing import Any

class AttribEvaluator:
    '''
    Evaluates attribute prediction performance for a given attribute.
    Calculates precision, recall, and f1 score for each class, as well as weighted averages.

    Can also calculate overall metrics (i.e. simulated usage of the model) for the dataset.
    Calculates false positives, false negatives, and mean labels per image.
    And also precision, recall, and f1 score similar to above.
    '''
    
    def __init__(self, target_attribute, attrib_mapping, calc_overall_metrics=False, iou_threshold=0.5, score_threshold=0.7):
        self.target_attribute = target_attribute
        self.iou_threshold = iou_threshold # minimum iou required to count as a positive match
        self.score_threshold = score_threshold # minimum score required to count as a positive match

        self.attrib_scores = []
        self.gt_labels = []
        self.pred_labels = []
        
        self.attrib_mapping = attrib_mapping
        self.num_classes = len(attrib_mapping[target_attribute])
        self.missed_gts = torch.zeros(self.num_classes, dtype=torch.int32)
        
        self.calc_overall_metrics = calc_overall_metrics
        if calc_overall_metrics:
            self.false_pos = 0
            self.false_neg = 0
            self.num_images = 0
            self.overall_gt_labels = []
            self.overall_pred_labels = []
            self.gt_label_count = 0

    # add a set of results, typically from one batch, to the evaluator
    def update(self, results, targets):
        # example of results[0]
        # {
        #     'boxes': tensor([[ 605.4993,   37.1151, 1192.0000,  384.9551], [ 651.5644,    0.0000,  950.7262,  576.1069],
        #             ... [ 507.8282,  425.5512, 1192.0000,  730.0000], [   0.0000,    0.0000,  427.4406,  398.8574]]), 
        #     'labels': tensor([1, 1, ..., 1, 1]), 
        #     'scores': tensor([0.3550, 0.3504, ..., 0.2879, 0.2866]), 
        #     'attributes':  {
        #         'category': {
        #             'scores': tensor([0.3874, 0.3921, ..., 0.3970, 0.4153]),
        #             'labels': tensor([2, 2, ..., 2, 2])
        #         }
        #     }
        # }

        # example of targets[0]
        # {
        #     'boxes': tensor([[ 262.3500,   65.1500, 1067.1300,  521.6600]]), 
        #     'labels': tensor([1]), 
        #     'image_id': 764, 
        #     'area': tensor([367390.1250]), 
        #     'iscrowd': tensor([0]), 
        #     'attributes': {
        #         'category': tensor([2])
        #     }
        # }

        # calculate attribute prediction performance
        # i.e. how good is attribute prediction when the object is detected
        # for each box in targets
        #   calculate the iou with all boxes in results
        #   find the box with the highest iou
        #   if iou > iou_threshold and label matches, then it is a true positive
        #   if iou < iou_threshold or label does not match, then it is a false positive

        # n = sum([t['labels'].shape[0] for t in targets])
        gt_labels = torch.cat([t['attributes'][self.target_attribute] for t in targets])

        # ignore the boxes which are not labeled
        gt_labels = gt_labels[gt_labels != 0]

        n = gt_labels.shape[0]
        attrib_scores = torch.zeros(n, dtype=torch.float32)
        pred_labels = torch.zeros(n, dtype=torch.int32)
        
        i = 0
        assert len(results)==len(targets), "LENGTH OF RESULTS != LENGTH OF TARGETS"

        # iterate through result/target for each image
        for result, target in zip(results, targets):
            # iterate through ground truths for that image
            assert len(target['boxes']) == len(target['attributes'][self.target_attribute]), "number of boxes != number of attribute labels"
            for box, label in zip(target['boxes'], target['attributes'][self.target_attribute]):
                if label == 0:
                    continue

                max_iou = 0
                max_iou_index = None

                # find prediction with highest iou with current gt
                for j, result_box in enumerate(result['boxes']):
                    iou = compute_iou(box, result_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_index = j

                # if that prediction has iou > threshold, it's a successful prediction
                if max_iou_index is not None and max_iou > self.iou_threshold: # and result['attributes'][self.target_attribute]['labels'][j] == label:
                    # save the label and score of that prediction
                    pred_labels[i] = result['attributes'][self.target_attribute]['labels'][max_iou_index]
                    attrib_scores[i] = result['attributes'][self.target_attribute]['scores'][max_iou_index]
                else:
                    self.missed_gts[label] += 1
                    # print(f"failed to match a prediction to this gt, max_iou_index: {max_iou_index}, max_iou: {max_iou}")
                    # print(f"image_id: {target['image_id']}, gt box: {box}, label: {label}")
                    # print(f"predicted boxes: {result['boxes']}")
                    pass # if it's not a successful prediction, pred_labels[i] will remain 0
                if gt_labels[i] != label:
                    raise ValueError("attrib evaluator not updating properly")
                i += 1

        if i!=len(attrib_scores):
            print(gt_labels)
            print(len(gt_labels))
            print(i)
            raise ValueError("number of gt labels processed is wrong")

        # save results of attribute prediction performance evaluation
        self.attrib_scores.append(attrib_scores)
        self.gt_labels.append(gt_labels)
        self.pred_labels.append(pred_labels)

        # calculate overall metrics
        # now iterate through predicted boxes in outer for loop, since we "don't know" the ground truths,
        # we treat the predicted boxes in order of score
        # filter out pred boxes below a score threshold
        # match each pred box with a gt box, using IoU threshold
        # pred with no match -> false positive
        # gt with no match -> false negative
        if self.calc_overall_metrics:
            self.num_images += len(results)

            overall_gt_labels = []
            overall_pred_labels = []

            for result, target in zip(results, targets):
                pred_boxes = result['boxes']
                pred_scores = result['scores']

                # filter out boxes above threshold
                mask = pred_scores > self.score_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]

                # sort scores and boxes by scores
                s = torch.argsort(pred_scores)
                pred_boxes = pred_boxes[s]

                self.gt_label_count += len(target['boxes']) # count true positives

                # mapping logic
                pred_gt_box_mapping = {}
                for i, pred_box in enumerate(pred_boxes): # iterate through predicted boxes
                    max_match_iou = -1
                    for j, gt_box in enumerate(target['boxes']): # for given pred box, iterate through gts to find best match
                        if j in pred_gt_box_mapping.values():
                            continue
                        iou = compute_iou(gt_box, pred_box)
                        # if iou threshold is met and iou is highest so far, assign pred box (i) to gt box (j)
                        if iou>self.iou_threshold and iou>max_match_iou: 
                            max_match_iou = iou
                            pred_gt_box_mapping[i] = j

                    # if pred box failed to be matched to any gts, it's a false positive
                    if i not in pred_gt_box_mapping.keys():
                        self.false_pos += 1
                    # else take its attribute data for comparison to matched gt
                    else:
                        matched_gt_index = pred_gt_box_mapping[i]
                        gt_label = target['attributes'][self.target_attribute][matched_gt_index]

                        # don't use the matched gt for attribute prediction evaluation if it's an unknown attribute
                        # we kept it around for matching so that pred boxes can still be mapped to buildings without attribute data
                        # to ensure our false positives/false negatives detections are more accurate
                        if gt_label != 0:
                            overall_gt_labels.append(gt_label)
                            overall_pred_labels.append(result['attributes'][self.target_attribute]['labels'][i])

                # check how many gts didn't get matched
                self.false_neg += (len(target['boxes']) - len(pred_gt_box_mapping))

            if overall_gt_labels:
                self.overall_gt_labels.append(torch.stack(overall_gt_labels))
                self.overall_pred_labels.append(torch.stack(overall_pred_labels))
            
            

    def evaluate(self, print_cm):
        
        print(f"*******evaluating {self.target_attribute}*******")
        print("===========ATTRIBUTE PREDICTION METRICS===========")
        print("missed gts by class:", self.missed_gts)
        if self.pred_labels and self.gt_labels:
            predicted = torch.cat(self.pred_labels) #torch.cat(dt[attrib_name])
            targets = torch.cat(self.gt_labels) #torch.cat(gt[attrib_name])
            n = self.num_classes
            wavg_f1 = calculate_metrics(predicted, targets, n, print_cm)
        else:
            print("NOTHING TO EVALUATE")
            wavg_f1 = 0

        if self.calc_overall_metrics:
            print("===========OVERALL METRICS===========")
            if self.overall_gt_labels and self.overall_pred_labels:
                print(f"Mean labels per image: {self.gt_label_count/self.num_images:.3f}")
                print(f"Mean false negatives per image: {self.false_neg/self.num_images:.3f}")
                print(f"Mean false positives per image: {self.false_pos/self.num_images:.3f}")
                overall_predicted = torch.cat(self.overall_pred_labels)
                overall_targets = torch.cat(self.overall_gt_labels)
                overall_wavg_f1 = calculate_metrics(overall_predicted, overall_targets, n, print_cm)
                return torch.tensor([wavg_f1, overall_wavg_f1])
            else:
                print("NOTHING TO EVALUATE")
                
        return torch.tensor(wavg_f1)

def calculate_metrics(predicted, targets, n, print_cm):
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

    if print_cm:
        # calculate confusion matrix given ground truth and predicted labels
        confusion_matrix = torch.zeros(n, n, dtype=torch.int32)
        for i, j in zip(targets, predicted):
            confusion_matrix[i, j] += 1

        # print the confusion matrix
        print_confusion_matrix(confusion_matrix, class_labels=None)

    return wavg_f1



def compute_iou(bbox_1, bbox_2):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    # bbox_1 and bbox_2 are tensors of shape (4,), and have format xyxy
    # returns the intersection over union of bbox_1 and bbox_2
    assert bbox_1.shape == (4,) and bbox_2.shape == (4,), "Boxes used for computing IoU don't have shape (4,)"

    # Calculate the coordinates of the intersection rectangle
    x1 = torch.max(bbox_1[0], bbox_2[0])
    y1 = torch.max(bbox_1[1], bbox_2[1])
    x2 = torch.min(bbox_1[2], bbox_2[2])
    y2 = torch.min(bbox_1[3], bbox_2[3])

    # Calculate the area of intersection rectangle
    intersection_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    # Calculate the area of both bounding boxes
    bbox_1_area = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1)
    bbox_2_area = (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1)

    # Calculate the union area by subtracting the intersection area
    # and adding the areas of both bounding boxes
    union_area = bbox_1_area + bbox_2_area - intersection_area

    if union_area == 0:
        return torch.tensor(0.)

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def print_confusion_matrix(confusion_matrix, class_labels=None):
    # If no labels are provided, use numeric labels
    if class_labels is None:
        class_labels = [str(i) for i in range(len(confusion_matrix))]
    
    # Determine the maximum number of digits in any entry of the confusion matrix
    max_digits = max(count_digits(number) for row in confusion_matrix for number in row)
    
    # Determine the maximum width needed for any column by comparing label and value widths
    max_label_length = max(max([len(label) for label in class_labels]), max_digits)
    
    # Format strings for the header and rows, ensuring alignment
    header_format = ("{:>" + str(max_label_length) + "} ") * (len(class_labels) + 1)
    row_format = "{:>" + str(max_label_length) + "} " + ("{:>" + str(max_label_length) + "} ") * len(class_labels)
    
    # Print the header
    print(" " * (max_label_length + 3) + "Predicted Classes")
    print(header_format.format("", *class_labels))
    
    # Print each row of the matrix
    for i, row in enumerate(confusion_matrix):
        print(row_format.format(class_labels[i], *row.tolist()))


def count_digits(n):
    if n > 0:
        return int(math.log10(n))+1
    elif n == 0:
        return 1
    else:
        return int(math.log10(-n))+2 # +1 if you don't count the '-' 