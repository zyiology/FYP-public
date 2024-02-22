# from faster_rcnn
import torch.nn.functional as F
import torchvision
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.roi_heads import RoIHeads
from typing import Any
# from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights

# from generalized_rcnn
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional
import torch
from torch import nn, Tensor

# from roi_heads
from torchvision.ops import boxes as box_ops

# from engine
import utils
import math
import sys
import time
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

from torchvision.models.detection.roi_heads import keypointrcnn_loss, keypointrcnn_inference
from torchvision.models.detection.faster_rcnn import FasterRCNN

from torchvision.models.detection import fasterrcnn_resnet50_fpn

# need to modify roiheads?
class CustomFasterRCNN(FasterRCNN):

    def __init__(self, backbone, num_classes=None, num_attributes=None, attribute_weights=None, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        # self.roi_heads = None # define my own?

        # taken from faster_rcnn.py
        representation_size = 1024

        # Replace roi_heads with your own MyRoiHead instance
        # Use the attributes initialized by the FasterRCNN constructor
        self.roi_heads = MyRoIHead(
            attribute_weights=attribute_weights,
            box_roi_pool=self.roi_heads.box_roi_pool,
            box_head=self.roi_heads.box_head,
            box_predictor=CustomFastRCNNPredictor(representation_size, num_classes, num_attributes),
            fg_iou_thresh=self.roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=self.roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=self.roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=self.roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=self.roi_heads.box_coder.weights,
            score_thresh=self.roi_heads.score_thresh,
            nms_thresh=self.roi_heads.nms_thresh,
            detections_per_img=self.roi_heads.detections_per_img,
            # ... other arguments as needed ...
        )

    # from generalized_rcnn
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


model_urls = {
    "fasterrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
}


def custom_fasterrcnn_resnet50_fpn(
    pretrained=False, progress=True, num_classes=91, num_attribs=5, attribute_weights=None, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=progress,
                        norm_layer=misc_nn_ops.FrozenBatchNorm2d)  # pretrained=pretrained_backbone,
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = CustomFasterRCNN(backbone, num_classes, num_attribs, attribute_weights, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


class MyRoIHead(RoIHeads):
    def __init__(self, attribute_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute_weights = attribute_weights
        # can add extra params if needed here

    # features is a dictionary, each value is a tensor of features from 1 image
    def forward(self, features, proposals, image_shapes, targets=None):
        # self.check_targets(targets)

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                assert t["attributes"].dtype == torch.int64, "target attributes must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

        if self.training:
            proposals, matched_idxs, labels, attributes, regression_targets = (
                self.select_training_samples(proposals, targets))
        else:
            labels = None
            attributes = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)

        # need to make sure box_predictor is a CustomFastRCNNPredictor
        class_logits, attribute_logits, box_regression = self.box_predictor(box_features)
        # print("class logits:", class_logits)
        # print("attrib logits:", attribute_logits)

        # new feature here, using potentially box_features

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and attributes is not None and regression_targets is not None
            loss_classifier, loss_box_reg, loss_attribute = (
                fastrcnn_loss_with_attributes(class_logits, attribute_logits, box_regression,
                                              labels, attributes, self.attribute_weights, regression_targets))
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg,
                      "loss_attribute": loss_attribute}
            #print(losses)
        else:
            boxes, scores, labels, attribute_score, attribute_label = self.postprocess_detections_with_attributes(class_logits,
                                                                                            attribute_logits,
                                                                                            box_regression,
                                                                                            proposals,
                                                                                            image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "attribute_score": attribute_score[i],
                        "attribute_label": attribute_label[i]
                    }
                )

        # probably not important for my implementation, no keypoints
        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
                self.keypoint_roi_pool is not None
                and self.keypoint_head is not None
                and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_attributes = [t["attributes"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, attributes = (
            self.assign_targets_to_proposals_with_attributes(proposals, gt_boxes, gt_labels, gt_attributes))
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            attributes[img_id] = attributes[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, attributes, regression_targets

    def assign_targets_to_proposals_with_attributes(self, proposals: List[Tensor], gt_boxes: List[Tensor],
                                                    gt_labels: List[Tensor], gt_attributes: List[Tensor]
                                                    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        matched_idxs = []
        labels = []
        attributes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_attrib_in_image in (
                zip(proposals, gt_boxes, gt_labels, gt_attributes)):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                attrib_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                attrib_in_image = gt_attrib_in_image[clamped_matched_idxs_in_image]
                attrib_in_image = attrib_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                attrib_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                attrib_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            attributes.append(attrib_in_image)
        return matched_idxs, labels, attributes

    def postprocess_detections_with_attributes(
            self,
            class_logits,  # type: Tensor
            attribute_logits,  # type: Tensor  # Add attribute logits as an input
            box_regression,  # type: Tensor
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_attributes = F.softmax(attribute_logits, -1)  # Apply softmax to attribute logits

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_attributes_list = pred_attributes.split(boxes_per_image, 0)  # Split attribute predictions

        all_boxes = []
        all_scores = []
        all_labels = []
        all_attribute_score = []  # List to store all attribute scores
        all_attribute_label = []  # List to store all attribute labels

        # each iteration corresponds to per-image processing
        for boxes, scores, attributes, image_shape in zip(pred_boxes_list, pred_scores_list, pred_attributes_list,
                                                          image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # find the attribute with highest probability that's not background
            # its index is equal to its index in the attribute logits + 1
            attribute_label = torch.argmax(attributes[:, 1:], dim=1) + 1

            # get the score of the attribute with the highest probability
            # The torch.max function actually returns a tuple of two tensors: the maximum values and the indices of the maximum values. By adding [0] at the end of the statement, we are selecting only the first element of the tuple, which is the tensor of maximum values.
            attribute_score = torch.max(attributes[:, 1:], dim=1)[0]

            # squeeze attribute_label and attribute_score into one column, then repeat by num_classes
            # so that each box will have an associated attribute label and score, that can be processed in the same way as the class labels and scores
            attribute_label = attribute_label.unsqueeze(1).repeat(1, num_classes)
            attribute_score = attribute_score.unsqueeze(1).repeat(1, num_classes)

            # process attribute predictions
            # Assuming one attribute per box, reshape if needed
            # attributes = attributes.reshape(-1, num_attributes)

            # Remove predictions with the background label
            # each row represents a proposal, each column represents a class
            # for each proposal, generated N boxes and N scores, one for each class
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            attribute_label = attribute_label[:, 1:]
            attribute_score = attribute_score[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            attribute_label = attribute_label.reshape(-1)
            attribute_score = attribute_score.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, attribute_label, attribute_score = boxes[inds], scores[inds], labels[inds], attribute_label[inds], attribute_score[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, attribute_label, attribute_score = boxes[keep], scores[keep], labels[keep], attribute_label[keep], attribute_score[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, attribute_label, attribute_score = boxes[keep], scores[keep], labels[keep], attribute_label[keep], attribute_score[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_attribute_score.append(attribute_score)  # Append attributes for each image
            all_attribute_label.append(attribute_label) 

        return all_boxes, all_scores, all_labels, all_attribute_score, all_attribute_label  # Return attributes along with other detections


class CustomFastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, num_attributes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.attribute_pred = nn.Linear(in_channels, num_attributes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        attributes = self.attribute_pred(x)

        return scores, attributes, bbox_deltas


# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()

def fastrcnn_loss_with_attributes(class_logits, attribute_logits, box_regression, labels, attributes, attribute_weights, regression_targets):
    # type: (Tensor, Tensor, Tensor, List[Tensor], List[Tensor], Tensor, List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        attribute_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        attributes (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    attributes = torch.cat(attributes, dim=0)

    # get indices of attributes that are not background
    attributes_nonzero = torch.nonzero(attributes, as_tuple=True)[0] # returns a tuple of tensors, get the first one

    # filter out the non-background attribute logits and labels
    attribute_logits = attribute_logits[attributes_nonzero, :]
    attributes = attributes[attributes_nonzero]

    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # custom weighted cross entropy
    # attribute_loss = (F.cross_entropy(attribute_logits, attributes, weight=attribute_weights, reduction='sum') /
    #                   len(attributes))

    # focal loss

    # alpha = attribute_weights
    alpha = 0.25
    gamma = 2
    ce_loss = F.cross_entropy(attribute_logits, attributes, reduction='none')
    pt = torch.exp(-ce_loss)

    if (type(alpha) == torch.Tensor and len(alpha)>1):
        # Index alpha with the class labels to get a tensor of size N
        alpha_selected = alpha[attributes]
        # Ensure focal_factor is broadcastable with ce_loss
        # ce_loss is of size N, so we squeeze focal_factor to match this shape
        focal_factor = (1 - pt) ** gamma
        focal_factor = focal_factor.squeeze()

        # Compute the focal loss
        focal_loss = alpha_selected * focal_factor * ce_loss
    else:
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    # Average the loss over the batch
    attribute_loss = focal_loss.mean()

    # attribute_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch

    # print('attribute logits:', attribute_logits)
    # print('attributes:', attributes)
    # print('labels:', labels)
    # print('attribute loss:', attribute_loss)

    # probs = F.softmax(attribute_logits, dim=1)
    #
    # gamma = 2.0  # Focusing parameter
    # alpha = attribute_weights # Example constant value e.g. 0.25, or use a tensor for class-dependent values
    #
    # # Compute the focal loss
    # # Gather the probabilities of the target classes
    # target_probs = probs.gather(1, attributes.unsqueeze(1))
    # # Calculate the focal loss factor
    # focal_factor = (1 - target_probs) ** gamma
    # # Calculate the cross-entropy loss without reduction
    # cross_entropy_loss = F.cross_entropy(attribute_logits, attributes, reduction='none')
    # # Apply the focal factor and alpha to the cross-entropy loss
    # focal_loss = alpha * focal_factor * cross_entropy_loss
    # # Sum or average the losses
    # attribute_loss = focal_loss.mean()

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss, attribute_loss

# yet to be fixed
@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator = AttribEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    if isinstance(model_without_ddp, CustomFasterRCNN):
        iou_types.append("attrib")
    return iou_types

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


class AttribEvaluator(CocoEvaluator):
    def __init__(self, dataset_name, iou_types):
        super().__init__(dataset_name, iou_types)
    
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        if iou_type == "attrib":
            return self.prepare_for_coco_attrib(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")
    
    def prepare_for_coco_attrib(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            attrib_scores = prediction["attribute_score"].tolist()
            attrib_labels = prediction["attribute_label"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": attrib_labels[k],
                        "bbox": box,
                        "score": attrib_scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results




def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
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
        )# work out when's the right time to send this to device.

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        # print(targets)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# @register_model()
# @handle_legacy_interface(
#     weights=("pretrained", FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
#     weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
# )


# def custom_fasterrcnn_resnet50_fpn(
#     *,
#     weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
#     progress: bool = True,
#     num_classes: Optional[int] = None,
#     weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
#     trainable_backbone_layers: Optional[int] = None,
#     **kwargs: Any,
# ) -> CustomFasterRCNN:
#     weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
#     weights_backbone = ResNet50_Weights.verify(weights_backbone)
#
#     if weights is not None:
#         weights_backbone = None
#         num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
#     elif num_classes is None:
#         num_classes = 91
#
#     is_trained = weights is not None or weights_backbone is not None
#     trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
#     norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d
#
#     backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
#     backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
#     model = CustomFasterRCNN(backbone, num_classes=num_classes, **kwargs)
#
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
#         if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
#             overwrite_eps(model, 0.0)
#
#     return model
#
# def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
#     if actual is not None:
#         if actual != expected:
#             raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
#     return expected