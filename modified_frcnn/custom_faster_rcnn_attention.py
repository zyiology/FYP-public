import utils
import math
import sys
import time
import copy
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional
from os.path import join
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.io import read_image
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import boxes as box_ops
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.roi_heads import RoIHeads, keypointrcnn_loss, keypointrcnn_inference
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

# from local files
from engine import _get_iou_types
from attrib_eval import AttribEvaluator
from coco_eval import CocoEvaluator

# main class for modified Faster R-CNN
class CustomFasterRCNN(FasterRCNN):

    # num_classes can technically be any number > 2, but was generally tested with 2 (i.e. building or not building)
    # attribute_mappings is a dictionary of dictionaries that map attribute classes to their corresponding indices
    # attribute_weights_dict is a dictionary of dictionaries that map attribute indices to their corresponding weights
    # num_heads is the number of heads for the multihead attention layer
    # use_attention is a boolean that determines whether attention is used (i.e. modification 3 in the paper)
    # attention_per_attrib is a boolean that determines whether separate self-attention are applied per attribute (vs sharing one)
    # custom_box_predictor is a boolean that determines whether self-attention is used for the box predictor
        # the default box_predictor is a FastRCNNPredictor, defined in pytorch
    # use_reduced_features_for_attrib is a boolean that determines whether reduced features are used for attribute prediction
        # i.e. modification 1 vs 2 in the paper
    # parallel backbone is a boolean that determines whether a parallel backbone is used for attribute prediction
        # i.e. modification 4 in the paper

    def __init__(self, backbone, num_classes=None, attribute_mappings=None, attribute_weights_dict=None, 
                 num_heads = 2, use_attention=True, attention_per_attrib=True, custom_box_predictor=False,  
                 use_reduced_features_for_attrib=False, parallel_backbone=None, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)

        channels = backbone.out_channels

        if attribute_mappings == None:
            raise ValueError("attributes_dims is None")
  
        # default values for output from RoI pooling
        height, width = 7, 7
        resolution = self.roi_heads.box_roi_pool.output_size[0]
        representation_size = 1024

        attributes_head_params = {"resolution": resolution,
                                  "representation_size": representation_size,
                                  "out_channels": channels,
                                  "num_heads": num_heads,
                                  "height": height,
                                  "width": width}

        # Replace roi_heads with CustomRoIHead instance, which incorporates attribute prediction
        # Use the attributes initialized by the FasterRCNN constructor
        self.roi_heads = CustomRoIHead(
            attribute_weights_dict=attribute_weights_dict,
            attribute_predictor=AttributesPredictor(
                # channels=channels,
                # num_heads=num_heads,
                attribute_mappings=attribute_mappings,
                # height=height,
                # width=width,
                use_attention = use_attention,
                attention_per_attrib=attention_per_attrib,
                attributes_head_params = attributes_head_params,
                use_reduced_features_for_attrib=use_reduced_features_for_attrib),
            use_reduced_features_for_attrib=use_reduced_features_for_attrib,
            box_roi_pool=self.roi_heads.box_roi_pool,
            box_head=self.roi_heads.box_head,
            box_predictor=self.roi_heads.box_predictor,
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

        if custom_box_predictor:
            self.roi_heads.box_predictor = CustomFastRCNNPredictor(channels, num_heads, height, width, num_classes, hidden_dim=1024)

        self.parallel_backbone = parallel_backbone

    # modified from generalized_rcnn
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]], bool) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (dict[str, Tensor] or list[dict[str, Tensor]]): the output from the model.
                During training, it returns a dict[str, Tensor] which contains the losses.
                During testing, it returns list[dict[str, Tensor]], where each dict contains the results for an image
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

        # resizes the images to fit faster r-cnn (defined in base class)
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
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

        # pass image tensors through CNN backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # pass image tensors through parallel CNN backbone if it exists
        if self.parallel_backbone:
            parallel_features = self.parallel_backbone(images.tensors)
            if isinstance(parallel_features, torch.Tensor):
                parallel_features = OrderedDict([("0", parallel_features)])   

        # pass feature maps through region proposal network
        proposals, proposal_losses = self.rpn(images, features, targets)

        # extract proposals from feature map to pass to roi_heads
        if self.parallel_backbone:
            detections, detector_losses = self.roi_heads(parallel_features, proposals, images.image_sizes, targets)
        else:
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # transform detections based on original image sizes
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
            # return losses for training and detections for inference
            return self.eager_outputs(losses, detections)

# Faster R-CNN models with different backbones
model_urls = {
    "fasterrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
}

# adapted from fasterrcnn_resnet50_fpn in torchvision
def custom_fasterrcnn_resnet_fpn(
    pretrained=False, progress=True, num_classes=2, attrib_mappings=None, attribute_weights_dict=None, pretrained_backbone=True, trainable_backbone_layers=None, num_heads = 2,
    attention_per_attrib=True, custom_box_predictor=False, use_reduced_features_for_attrib=False, use_attention=True, parallel_backbone=False, 
    use_resnet101=False, **kwargs
):
    """
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
        - attributes (``Dict[str, Int64Tensor[N]]``): a dictionary of attribute labels for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN. The R-CNN has attribute losses as well.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection
        - attributes (``Dict[str, Dict[str, Tensor[N]]``): a dictionary of attribute scores and labels for each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

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

    if attrib_mappings == None:
        raise ValueError("need to map attributes")
    
    if pretrained_backbone:
        weights = ResNet101_Weights.IMAGENET1K_V1 if resnet101 else ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    resnet = resnet101 if use_resnet101 else resnet50

    backbone = resnet(weights=weights, 
                      progress=progress, 
                      norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    if parallel_backbone:
        parallel_backbone = copy.deepcopy(backbone)

    model = CustomFasterRCNN(backbone, 
                             num_classes=num_classes, 
                             attribute_mappings=attrib_mappings, 
                             attribute_weights_dict=attribute_weights_dict,
                             attention_per_attrib=attention_per_attrib, 
                             custom_box_predictor=custom_box_predictor, 
                             use_attention=use_attention, 
                             num_heads=num_heads,
                             use_reduced_features_for_attrib=use_reduced_features_for_attrib, 
                             parallel_backbone=parallel_backbone, 
                             **kwargs)
    
    if pretrained:
        # should modify this to account for resnet101, there's no state dict for resnet101 version
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)

    return model

# adapted from RoIHeads from torchvision
class CustomRoIHead(RoIHeads):
    def __init__(self, attribute_weights_dict=None, attribute_predictor=None, 
                 self_attention=None, use_reduced_features_for_attrib=False, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute_weights_dict = attribute_weights_dict
        self.attribute_predictor = attribute_predictor
        self.self_attention = self_attention

        if isinstance(self.box_predictor, CustomFastRCNNPredictor) and use_reduced_features_for_attrib:
            raise ValueError("if using custom box predictor, reduced features will not be computed")

        self.use_reduced_features_for_attrib = use_reduced_features_for_attrib

    # features is a dictionary, each value is a tensor of features from 1 image
    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                for attr_value in t["attributes"].values():
                    assert attr_value.dtype == torch.int64, "target attributes must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

        # if in training mode, match proposals with ground truth boxes to get target class/attribute classes
        # attribute_dicts is a list of dictionaries, type List[Dict[str,Tensor]]

        if self.training:      
            proposals, matched_idxs, labels, attribute_dicts, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            attribute_dicts = None
            regression_targets = None
            matched_idxs = None
        
        # output of this, by default, is 7*7 (defined in faster_rcnn)*out_channels of backbone (resnet50_fpn)
        # could be increased to provide more features for attribute prediction
        box_features = self.box_roi_pool(features, proposals, image_shapes) 

        # if using custom_box_predictor, send proposal tensor directly to box_predictor
        if isinstance(self.box_predictor, CustomFastRCNNPredictor):
            class_logits, box_regression = self.box_predictor(box_features)
        else:
            reduced_box_features = self.box_head(box_features) # this is the step that flattens the feature vector
            class_logits, box_regression = self.box_predictor(reduced_box_features)

        # attributes_logits_dict will look something like {"material":tensor([0.1,0.2,0.3,0.1]), "type":tensor([0.02,0.04])}
        if self.use_reduced_features_for_attrib:
            attributes_logits_dict = self.attribute_predictor(reduced_box_features) 
        else:
            attributes_logits_dict = self.attribute_predictor(box_features) 

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training: # calculate losses
            assert labels is not None and attribute_dicts is not None and regression_targets is not None
            loss_classifier, loss_box_reg, loss_attribute = (
                fastrcnn_loss_with_attributes(class_logits, attributes_logits_dict, box_regression,
                                              labels, attribute_dicts, self.attribute_weights_dict, regression_targets))
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg,
                      "loss_attribute": loss_attribute}
        else:
            boxes, scores, labels, attribute_score_dict, attribute_label_dict = self.postprocess_detections_with_attributes(
                class_logits, attributes_logits_dict, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            # repack results so it's formatted as one dictionary per image
            for i in range(num_images):
                current = {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "attributes": {key: {"scores": attribute_score_dict[key][i], "labels":attribute_label_dict[key][i]}
                     for key in attribute_label_dict.keys()}
                }
                result.append(current)

        # probably not important for my implementation, no keypoints
        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        # if (
        #         self.keypoint_roi_pool is not None
        #         and self.keypoint_head is not None
        #         and self.keypoint_predictor is not None
        # ):
        #     keypoint_proposals = [p["boxes"] for p in result]
        #     if self.training:
        #         # during training, only focus on positive boxes
        #         num_images = len(proposals)
        #         keypoint_proposals = []
        #         pos_matched_idxs = []
        #         assert matched_idxs is not None
        #         for img_id in range(num_images):
        #             pos = torch.where(labels[img_id] > 0)[0]
        #             keypoint_proposals.append(proposals[img_id][pos])
        #             pos_matched_idxs.append(matched_idxs[img_id][pos])
        #     else:
        #         pos_matched_idxs = None

        #     keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        #     keypoint_features = self.keypoint_head(keypoint_features)
        #     keypoint_logits = self.keypoint_predictor(keypoint_features)

        #     loss_keypoint = {}
        #     if self.training:
        #         assert targets is not None
        #         assert pos_matched_idxs is not None

        #         gt_keypoints = [t["keypoints"] for t in targets]
        #         rcnn_loss_keypoint = keypointrcnn_loss(
        #             keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
        #         )
        #         loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
        #     else:
        #         assert keypoint_logits is not None
        #         assert keypoint_proposals is not None

        #         keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
        #         for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
        #             r["keypoints"] = keypoint_prob
        #             r["keypoints_scores"] = kps

        #     losses.update(loss_keypoint)

        return result, losses

    # selects an equal proportion of positive/negative proposals to avoid class imbalance
    # modified from original RoIHeads to retrieve attribute classes from the ground truth
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
        matched_idxs, labels, attribute_dicts = (
            self.assign_targets_to_proposals_with_attributes(proposals, gt_boxes, gt_labels, gt_attributes))

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            for key, value in attribute_dicts[img_id].items():
                attribute_dicts[img_id][key] = value[img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, labels, attribute_dicts, regression_targets
    
    # modified from original RoIHeads to include attributes
    def assign_targets_to_proposals_with_attributes(self, proposals: List[Tensor], gt_boxes: List[Tensor],
                                                    gt_labels: List[Tensor], gt_attributes: List[Dict[str, Tensor]]
                                                    ) -> Tuple[List[Tensor], List[Tensor], List[Dict[str, Tensor]]]:

        matched_idxs = []
        labels = []
        attributes = []
        attribute_types = gt_attributes[0].keys()
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_attrib_in_image in (
                zip(proposals, gt_boxes, gt_labels, gt_attributes)):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                attrib_in_image = {at: torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device) for at in attribute_types}
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands

                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                attrib_in_image = {k: v[clamped_matched_idxs_in_image] for k, v in gt_attrib_in_image.items()}
                attrib_in_image = {k: v.to(dtype=torch.int64) for k,v in attrib_in_image.items()}

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                for value in attrib_in_image.values():
                    value[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                for value in attrib_in_image.values():
                    value[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            attributes.append(attrib_in_image)
        return matched_idxs, labels, attributes

    # process the raw logits for each proposal to get the final class and attribute probabilities
    def postprocess_detections_with_attributes(
            self,
            class_logits,  # type: Tensor
            attribute_logits,  # type: Dict[str,Tensor]
            box_regression,  # type: Tensor
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # softmax the logits to obtain probabilities
        pred_scores = F.softmax(class_logits, -1)
        pred_attributes = {key:F.softmax(val, -1) for key,val in attribute_logits.items()}  # Apply softmax to attribute logits

        # split the outputs per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_attributes_list_dict = {key:val.split(boxes_per_image, 0) for key,val in pred_attributes.items()}  # Split attribute predictions

        all_boxes = []
        all_scores = []
        all_labels = []
        all_attribute_score_dict = {key:[] for key in pred_attributes_list_dict.keys()}  # store all attribute scores
        all_attribute_label_dict = {key:[] for key in pred_attributes_list_dict.keys()}  # store all attribute labels

        # each iteration processes one image
        for idx, (boxes, scores, image_shape) in enumerate(zip(pred_boxes_list, pred_scores_list, image_shapes)):
            
            attributes_dict = {key:val[idx] for key,val in pred_attributes_list_dict.items()}
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # find the attribute with highest probability that's not Unknown/0, which is the predicted attribute
            # its index is equal to its index in the attribute logits + 1
            attribute_label_dict = {key:torch.argmax(val[:, 1:], dim=1) + 1 for key,val in attributes_dict.items()}

            # get the score/probability of the attribute with the highest probability
            # The torch.max function actually returns a tuple of two tensors: 
            # the maximum values and the indices of the maximum values. 
            # By adding [0] at the end of the statement, we are selecting only 
            # the first element of the tuple, which is the tensor of maximum values.
            attribute_score_dict = {key:torch.max(val[:, 1:], dim=1)[0] for key,val in attributes_dict.items()}

            # there is one box per class per proposal, but only one attribute prediction per proposal
            # so to ensure each box has a corresponding attribute, have to repeat by number of classes
            # refer to documentation for extended explanation
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key].unsqueeze(1).repeat(1, num_classes)
                attribute_score_dict[key] = attribute_score_dict[key].unsqueeze(1).repeat(1, num_classes)

            # Remove predictions with the background label
            # each row represents a proposal, each column represents a class (boxes has 3rd dimension for box coordinates)
            # for each proposal, generated N boxes and N scores, one for each class
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key][:, 1:]
                attribute_score_dict[key] = attribute_score_dict[key][:, 1:]

            # flatten scores and labels and attribute scores and labels, reshape boxes to be the same length in the 1st dimension
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key].reshape(-1)
                attribute_score_dict[key] = attribute_score_dict[key].reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels= boxes[inds], scores[inds], labels[inds]
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key][inds]
                attribute_score_dict[key] = attribute_score_dict[key][inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key][keep]
                attribute_score_dict[key] = attribute_score_dict[key][keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            for key in attribute_label_dict:
                attribute_label_dict[key] = attribute_label_dict[key][keep]
                attribute_score_dict[key] = attribute_score_dict[key][keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            for key in all_attribute_score_dict.keys():
                all_attribute_score_dict[key].append(attribute_score_dict[key])
                all_attribute_label_dict[key].append(attribute_label_dict[key]) 

        return all_boxes, all_scores, all_labels, all_attribute_score_dict, all_attribute_label_dict  # Return attributes along with other detections

# replaces the default class and bounding box predictor in Faster R-CNN
# implements a self-attention layer between the proposal tensor and the final classification and bounding box regression layers
class CustomFastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        channels (int): number of input channels
        num_heads (int): number of self-attention heads
        height, width (int): height and width of input feature map
        hidden_dim (int): number of hidden units in the feedforward layers
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, channels, num_heads, height, width, num_classes, hidden_dim):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.flattened_dim = height * width * channels
        self.attention = nn.MultiheadAttention(channels, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

    def forward(self, x):

        batch_size, c, h, w = x.size()
        assert c == self.channels and h == self.height and w == self.width, \
            f"Input tensor dimensions (channels, height, width) must be ({self.channels}, {self.height}, {self.width}), but got ({c}, {h}, {w})"

        seq_len = h * w
        input_dim = c

        # Reshape and permute `x` to shape (seq_len, batch, input_dim)
        x = x.permute(2, 3, 0, 1).reshape(seq_len, batch_size, input_dim)

        # add positional encodings to tensor before attention
        positional_encodings = positional_encoding_2d(h, w, c).to(x.device)
        positional_encodings = positional_encodings.view(h * w, 1, c).expand(-1, batch_size, -1)
        pe_x = x + positional_encodings

        # pass through self-attention layer, then feedforward layer
        attn_output, _ = self.attention(pe_x, pe_x, pe_x)
        flattened_output = attn_output.permute(1, 0, 2).contiguous().view(batch_size, -1)
        reduced_output = self.feed_forward(flattened_output)
        
        # pass through classification and bounding box regression layers
        scores = self.cls_score(reduced_output)
        bbox_deltas = self.bbox_pred(reduced_output)

        return scores, bbox_deltas
    
class AttributesPredictor(nn.Module): # channels, num_heads, height, width,
    def __init__(self, attribute_mappings, use_attention=True, attention_per_attrib=True, use_reduced_features_for_attrib=False, attributes_head_params=None):
        """
        channels (int): The number of channels in the input tensor.
        num_heads (int): Number of heads in the MultiheadAttention mechanism.
        attribute_mappings (dict[dict[str, int]]): A dictionary where keys are attribute names and values are the corresponding mappings.
        height, width (int): Dimensions of the spatial layout of the tensor before flattening.
        use_attention (bool): Whether to use self-attention for attribute prediction.
        attention_per_attrib (bool): Whether to use a separate attention layer for each attribute. If false, a single attention layer is used for all attributes.
        use_reduced_features_for_attrib (bool): Whether to use reduced features for attribute prediction. If true, don't pass the features through the TwoMLPHead module
        attributes_head_params (dict): Parameters for the TwoMLPHead module. Required if use_attention and use_reduced_features_for_attrib is false.
        """

        if use_reduced_features_for_attrib:
            assert not use_attention, "cannot use attention if reduced_features are used"

        super(AttributesPredictor, self).__init__()

        assert "out_channels" in attributes_head_params.keys(), "when instantiating AttributesPredictor, must provide out_channels in attributes_head_params"
        assert "height" in attributes_head_params.keys(), "when instantiating AttributesPredictor, must provide height in attributes_head_params"
        assert "width" in attributes_head_params.keys(), "when instantiating AttributesPredictor, must provide width in attributes_head_params"

        self.channels = attributes_head_params["out_channels"]
        self.height = attributes_head_params["height"]
        self.width = attributes_head_params["width"]
        self.flattened_dim = self.height * self.width * self.channels  # New flattened dimension
        self.reduced_dim = 1024 # could move this to be an input if so desired
        self.attention_per_attrib = attention_per_attrib
        self.use_attention = use_attention
        self.use_reduced_features_for_attrib = use_reduced_features_for_attrib
        
        if use_attention:
            assert "num_heads" in attributes_head_params.keys(), "if use_attention is true when instantiating AttributesPredictor, must provide num_heads in attributes_head_params"
            if attention_per_attrib:
                self.attentions = nn.ModuleDict({
                    attr: AttribAttention(self.channels, attributes_head_params["num_heads"], self.flattened_dim, self.reduced_dim) for attr in attribute_mappings.keys()
                })
            else:
                self.attention = AttribAttention(self.channels, attributes_head_params["num_heads"], self.flattened_dim, self.reduced_dim)
            self.attribute_predictors = nn.ModuleDict({
                attr: nn.Linear(self.reduced_dim, len(vals)) for attr, vals in attribute_mappings.items()
            })
            
            
        else:
            assert "representation_size" in attributes_head_params.keys(), "if use_attention is false is false when instantiating AttributesPredictor, must provide representation_size in attributes_head_params"
        
            representation_size = attributes_head_params["representation_size"]
            if not use_reduced_features_for_attrib:
                assert "resolution" in attributes_head_params.keys(), "if use_attention is false and use_reduced_features_for_attrib is false when instantiating AttributesPredictor, must provide resolution in attributes_head_params"
                assert "out_channels" in attributes_head_params.keys(), "if use_attention is false and use_reduced_features_for_attrib is false when instantiating AttributesPredictor, must provide out_channels in attributes_head_params"
                resolution, out_channels = attributes_head_params["resolution"], attributes_head_params["out_channels"]
                self.non_attn_heads = nn.ModuleDict({
                    attr: TwoMLPHead(out_channels * resolution**2, representation_size) for attr in attribute_mappings.keys()
                })

            self.attribute_predictors = nn.ModuleDict({
                attr: nn.Linear(representation_size, len(vals)) for attr, vals in attribute_mappings.items()
            })
            
        return

    def forward(
        self, 
        x # type: Tensor 
    ):
    # type: (...) -> Dict[str, Tensor]

        if self.use_attention:

            batch_size, c, h, w = x.size()
            
            assert c == self.channels and h == self.height and w == self.width, \
                f"Input tensor dimensions (channels, height, width) must be ({self.channels}, {self.height}, {self.width}), but got ({c}, {h}, {w})"

            seq_len = h * w
            input_dim = c

            # Reshape and permute `x` to shape (seq_len, batch, input_dim)
            x_reshaped = x.permute(2, 3, 0, 1).reshape(seq_len, batch_size, input_dim)

            # add positional encodings to tensor before
            positional_encodings = positional_encoding_2d(h, w, c).to(x.device)
            positional_encodings = positional_encodings.view(h * w, 1, c).expand(-1, batch_size, -1)

            pe_x = x_reshaped + positional_encodings

            if self.attention_per_attrib:
                attributes = {}

                for attr in self.attentions.keys():
                    attn_output = self.attentions[attr](pe_x)
                    attributes[attr] = self.attribute_predictors[attr](attn_output)
            else:
                attn_output = self.attention[attr](pe_x)
                attributes = {attr: predictor(attn_output) for attr, predictor in self.attribute_predictors.items()}
        else:
            # x should have shape (seq_len, batch, input_dim)
            attributes = {}

            if not self.use_reduced_features_for_attrib:
                for attr, head in self.non_attn_heads.items():
                    reduced_x = head(x)
                    attributes[attr] = self.attribute_predictors[attr](reduced_x)
                # attributes = {attr: predictor(reduced_x) for attr, predictor in self.attribute_predictors.items()}

            else:
                attributes = {attr: predictor(x) for attr, predictor in self.attribute_predictors.items()}

            # example of what variables look like
            # flattened_output.size(): torch.Size([2048, 12544])
            # x.size(): torch.Size([2048, 256, 7, 7])
            # attributes: {'roof_covers': tensor([[-0.8574, -0.3830, -0.2681, -0.3642, -0.0167, -0.4056],
            #     [-0.5766, -0.3567, -0.3244, -0.6630,  0.3329, -0.4042],
            #     [-0.4503, -0.5683, -0.8917, -0.2920,  0.4678, -0.0362],
            #     ...,
            #     [ 0.1379,  0.2471, -0.1716,  0.2814,  0.1945,  0.5834],
            #     [-0.3692, -0.2935, -0.1707, -0.3431,  0.5264, -0.3249],
            #     [ 0.0443,  0.0955,  0.2938,  0.2570,  0.2085, -0.1652]],
            # device='cuda:0', grad_fn=<AddmmBackward0>)}
        
        return attributes

class AttribAttention(nn.Module):
    def __init__(self, channels, num_heads, flattened_dim, output_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(flattened_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.flattened_dim = flattened_dim

    def forward(self, pe_x):
        attn_output, _ = self.attention(pe_x, pe_x, pe_x)

        # reshaping it from (seq_len, batch, input_dim) to (batch, seq_len*input_dim)
        flattened_output = attn_output.permute(1, 0, 2).reshape(attn_output.size(1), -1)

        assert flattened_output.shape[1] == self.flattened_dim, "flattened output of self-attention layer does not match input to feedforward network"

        return self.feed_forward(flattened_output)

# tested a modified TwoMLPHead that uses convolutional layers before the TwoMLPHead layers, but did not get improved performance - perhaps needs more work
class CNNandTwoMLPHead(nn.Module):
    """
    Extended head for FPN-based models incorporating CNN layers
    before the MLP layers for additional attribute predictions.

    Args:
        in_channels (int): number of input channels for the CNN layer (1024 in this case).
        representation_size (int): size of the intermediate representation for the MLP.
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()
        
        # Define the additional CNN layers
        # Example: One CNN layer followed by a ReLU activation function
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        # If adding another CNN layer, uncomment the following line
        # self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc6 = nn.Linear(512 * 7 * 7, representation_size) # Adjust the input size based on the output of your CNN layers
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # Apply the CNN layers with ReLU activation
        x = F.relu(self.conv1(x))
        # If you added a second CNN layer, apply it here
        # x = F.relu(self.conv2(x))
        
        # Flatten the output of the last CNN layer before passing it to the fully connected layers
        x = x.flatten(start_dim=1)
        
        # Apply the fully connected layers with ReLU activation
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

# modified loss function that returns attribute_loss, calculated using focal loss algorithm
def fastrcnn_loss_with_attributes(class_logits, attribute_logits_dict, box_regression, labels, attribute_dicts, attribute_weights_dict, regression_targets):
    # type: (Tensor, Dict[str, Tensor], Tensor, List[Tensor], List[Dict[str,Tensor]], Dict[str,Tensor], List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        attribute_logits_dict (Dict[str, Tensor])
        box_regression (Tensor)
        labels (list[Tensor])
        attributes (list[Dict[str,Tensor]])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        total_attribute_loss (Tensor)
    """

    device = labels[0].device

    labels = torch.cat(labels, dim=0)
    
    attribute_types = attribute_dicts[0].keys()
    for attrib_dict in attribute_dicts:
        assert attrib_dict.keys() == attribute_types, "All attribute dictionaries should have the same keys"

    # merge the dictionaries by key, concatenating the values
    attributes = {key: torch.cat([attrib_dict[key] for attrib_dict in attribute_dicts], dim=0) for key in attribute_types}

    # check if all attribute tensors have the same length
    attribute_lengths = [attribute.size(0) for attribute in attributes.values()]
    assert all(length == attribute_lengths[0] for length in attribute_lengths), "All attribute tensors should have the same length"

    # get indices of attributes that are not background, then filter out the non-background attribute logits and labels
    # do this for each type of attribute
    for key, val in attributes.items():
        val_nonzero = torch.nonzero(val, as_tuple=True)[0]
        attribute_logits_dict[key] = attribute_logits_dict[key][val_nonzero, :]
        attributes[key] = val[val_nonzero]

    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # calculate attribute loss, summing up for each attribute
    # focal loss
    # alpha = attribute_weights
    alpha = 0.25
    gamma = 2
    total_attribute_loss = torch.tensor(0., dtype=torch.float32).to(device)

    for key in attribute_types:
        # if attribute_weights_dict is defined, override the default alpha
        if attribute_weights_dict:
            alpha = attribute_weights_dict[key]

        attrib_logits = attribute_logits_dict[key]
        attrib = attributes[key]

        # Skip if attrib_logits and attrib are empty
        if attrib_logits.numel() == 0 and attrib.numel() == 0:
            continue

        ce_loss = F.cross_entropy(attrib_logits, attrib, reduction='none')
        if math.isnan(ce_loss):
            continue

        pt = torch.exp(-ce_loss)

        # if alpha is provided, calculate weighted focal loss
        # else alpha is same for all classes
        if (type(alpha) == torch.Tensor and len(alpha)>1):
            # Index alpha with the class labels to get a tensor of size N
            alpha_selected = alpha[attrib]

            # Compute the focal loss
            focal_factor = (1 - pt) ** gamma            
            focal_loss = alpha_selected * focal_factor * ce_loss

        else:
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        # Average the loss over the batch
        attribute_loss = focal_loss.mean()

        # if attribute_loss is nan, might have errors with focal loss - didn't need this while testing but theoretically could be a problem?
        if torch.isnan(attribute_loss):
            print("alpha", alpha)
            print("(1 - pt)", (1 - pt))
            print("ce_loss", ce_loss)
            print('attribute_loss', attribute_loss)

        total_attribute_loss += attribute_loss

    # get indices that correspond to the regression targets for the corresponding 
    # ground truth labels, to be used with advanced indexing
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

    return classification_loss, box_loss, total_attribute_loss

# evaluate the modified model
# modified from the original evaluate function in engine.py
# considered trying to calculate validation loss to determine when to stop training
# but need to modify rpn and roi_head to return both losses and detections to do this
@torch.inference_mode()
def evaluate(model, data_loader, device, eval_attrib=True, calc_overall_metrics=False, print_cm=False, score_threshold=0.7, iou_threshold=0.5):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = convert_to_coco_api(data_loader.dataset) #get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    print("iou_types:", iou_types)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    orig_dataset = get_dataset_from_dataloader(data_loader)

    if eval_attrib:
        attrib_evaluator_list = []
        for attrib_name in orig_dataset.attrib_mappings.keys():
            attrib_evaluator = AttribEvaluator(attrib_name, 
                                               attrib_mapping=orig_dataset.attrib_mappings, 
                                               iou_threshold=iou_threshold,
                                               score_threshold=score_threshold,
                                               calc_overall_metrics=calc_overall_metrics)
            attrib_evaluator_list.append(attrib_evaluator)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        with torch.no_grad():
            losses, outputs = model(images, targets, eager_output=False)

        # send to cpu for ease of processing
        def sendToCpu(output_dict):
            for key, value in output_dict.items():
                if key != "attributes":
                    output_dict[key] = value.to(cpu_device)
                else:
                    for attrib_key, v in value.items():
                        output_dict[key][attrib_key]['scores'] = v['scores'].to(cpu_device)
                        output_dict[key][attrib_key]['labels'] = v['labels'].to(cpu_device)
            return output_dict

        outputs = [sendToCpu(t) for t in outputs]
        model_time = time.time() - model_time

        # evaluate accuracy of detections
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)

        if eval_attrib:
            for attrib_evaluator, attrib_name in zip(attrib_evaluator_list, orig_dataset.attrib_mappings.keys()):
                attrib_evaluator.update(outputs, targets)
            
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if calc_overall_metrics:
        f1s = torch.zeros((len(attrib_evaluator_list), 2)) # each row in tensor contains (wavg_f1, overall_wavg_f1)
    else:
        f1s = torch.zeros((len(attrib_evaluator_list)))

    if eval_attrib:
        for i, attrib_evaluator in enumerate(attrib_evaluator_list):
            f1s[i] = attrib_evaluator.evaluate(print_cm=print_cm)

    torch.set_num_threads(n_threads)

    # consider returning attrib_evaluator
    return coco_evaluator, f1s


def get_dataset_from_dataloader(dataloader):
    dataset = dataloader
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset) or isinstance(dataset, torch.utils.data.DataLoader):
            dataset = dataset.dataset
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset
        
    raise ValueError("Could not find dataset in dataloader")


def modify_coco_attrib(coco_obj, attrib_name):
    coco_obj = copy.deepcopy(coco_obj)

    categories = set()

    for ann in coco_obj.dataset["annotations"]:
        ann["category_id"] = ann["attributes"][attrib_name]
        categories.add(ann["attributes"][attrib_name])

    coco_obj.dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_obj.createIndex()

    return coco_obj

def modify_results(results, attrib_name):
    results = copy.deepcopy(results)

    for img_id, output in results.items():

        # output["scores"] = output["attributes"][attrib_name]["scores"]
        output["labels"] = output["attributes"][attrib_name]["labels"]
        
    return results


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        attributes = targets["attributes"]
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            ann["attributes"] = {k:v[i].item() for k,v in attributes.items()}
            
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     if isinstance(model_without_ddp, CustomFasterRCNN):
#         iou_types.append("attrib")
#     return iou_types

# def convert_to_xywh(boxes):
#     xmin, ymin, xmax, ymax = boxes.unbind(1)
#     return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def positional_encoding_2d(H, W, D):
    if D % 2 != 0:
        raise ValueError("Depth D should be divisible by 2.")
    
    # Compute the positional encodings
    pe = torch.zeros((H, W, D), dtype=torch.float32)
    position_y = torch.arange(H).reshape(-1, 1, 1)
    position_x = torch.arange(W).reshape(1, -1, 1)
    div_term = torch.exp(torch.arange(0, D, 2).float() * -(torch.log(torch.tensor(10000.0)) / D))
    
    pe[:, :, 0::2] = torch.sin(position_y * div_term) + torch.sin(position_x * div_term)  # Even indices
    pe[:, :, 1::2] = torch.cos(position_y * div_term) + torch.cos(position_x * div_term)  # Odd indices
    
    return pe


# numpy version
# def positional_encoding_2d(H, W, D):
#     """
#     Generate a 2D positional encoding.
    
#     Parameters:
#     - H: Height of the tensor.
#     - W: Width of the tensor.
#     - D: Depth of the feature dimension, should be divisible by 2 for this encoding.
    
#     Returns:
#     - A tensor of shape [H, W, D] with positional encodings.
#     """
#     if D % 2 != 0:
#         raise ValueError("Depth D should be divisible by 2.")
    
#     # Compute the positional encodings
#     pe = np.zeros((H, W, D))
#     position_y = np.arange(H).reshape(-1, 1, 1)
#     position_x = np.arange(W).reshape(1, -1, 1)
#     div_term = np.exp(np.arange(0, D, 2) * -(np.log(10000.0) / D))
    
#     pe[:, :, 0::2] = np.sin(position_y * div_term) + np.sin(position_x * div_term)  # Even indices
#     pe[:, :, 1::2] = np.cos(position_y * div_term) + np.cos(position_x * div_term)  # Odd indices
    
#     pe = torch.tensor(pe, dtype=torch.float32)
#     return pe

# class AttribEvaluator(CocoEvaluator):
#     def __init__(self, dataset_name, iou_types):
#         super().__init__(dataset_name, iou_types)
    
#     def prepare(self, predictions, iou_type):
#         if iou_type == "bbox":
#             return self.prepare_for_coco_detection(predictions)
#         if iou_type == "segm":
#             return self.prepare_for_coco_segmentation(predictions)
#         if iou_type == "keypoints":
#             return self.prepare_for_coco_keypoint(predictions)
#         if iou_type == "attrib":
#             return self.prepare_for_coco_attrib(predictions)
#         raise ValueError(f"Unknown iou type {iou_type}")
    
#     def prepare_for_coco_attrib(self, predictions):
#         coco_results = []
#         for original_id, prediction in predictions.items():
#             if len(prediction) == 0:
#                 continue

#             boxes = prediction["boxes"]
#             boxes = convert_to_xywh(boxes).tolist()
#             attrib_scores = prediction["attribute_score"].tolist()
#             attrib_labels = prediction["attribute_label"].tolist()

#             coco_results.extend(
#                 [
#                     {
#                         "image_id": original_id,
#                         "category_id": attrib_labels[k],
#                         "bbox": box,
#                         "score": attrib_scores[k],
#                     }
#                     for k, box in enumerate(boxes)
#                 ]
#             )
#         return coco_results



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

        # need to specially handle attributes, because v is a Dict[str, Tensor]
        for t in targets:
            assert "attributes" in t.keys(), "target missing 'attributes' key"
            t["attributes"] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in t["attributes"].items()}

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

def eval_image(dataset, epoch, device, font_size=32,
                save=True, index=None, target_attrib=None, show_attrib=False, 
                model=None, subfolder=None, eval_transform=None, score_threshold=0.4):

    if show_attrib and not target_attrib:
        raise ValueError("Cannot show attribute without a target attribute")

    if index is None:
        index = dataset[0][1]['image_id']

    if eval_transform is None:
        transforms = []
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        eval_transform = T.Compose(transforms)

    image_tensor, target = dataset[index]
    image_tensor = image_tensor.to(device)

    image_id = target['image_id']

    # target = None # problem is full_dataset has transform where train=True, so some targets are randomly flipped
    # for i in range(len(dataset)):
    #     t = dataset.get_target(i)
    #     if t['image_id'] == index:
    #         target = t
    #         break
    # else:
    #     print('index does not exist in dataset')
    #     return 

    target_boxes = target['boxes']

    target_labels = [dataset.reverse_attrib_mappings[target_attrib][int(x)] for x in target['attributes'][target_attrib]]
    # attributes_strings = {}
    # for attrib_name, attrib_tensor in target['attributes'].items():
    #     attributes_strings[attrib_name] = [dataset.reverse_attrib_mappings[attrib_name][int(x)] for x in attrib_tensor]
    # #target_labels = [f"{full_dataset.material_classes_inverse[int(attrib_label)]}" for attrib_label in target['attributes']]

    # check if image exists
    possible_image_paths = [join(dataset.img_dir, f"image_{image_id}.png"),join(dataset.img_dir, f"image_{image_id}.jpg")] 
    
    i = 0
    while i<len(possible_image_paths):
        try:
            image = read_image(possible_image_paths[i])
            break
        except Exception: # maybe i should use a more specific exception?
            i += 1
            
    else:
        raise FileNotFoundError(f"image file for image id {target['image_id']} was not found")
        
    
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    output_image = draw_bounding_boxes(image, target_boxes, target_labels, colors="green", font='Roboto-Regular.ttf', font_size=font_size)
    
    if model:
    
        #eval_transform = get_transform(train=False)
        
        model.eval()
        with torch.no_grad():
            # x = eval_transform(image)
            # convert RGBA -> RGB and move to device
            # x = x[:3, ...].to(device)
            predictions = model([image_tensor, ])
            pred = predictions[0]

        for i, score in enumerate(pred['scores']):
            # prune off all the scores lower than the threshold
            if score<score_threshold:
                for key in pred.keys():
                    if key == "attributes":
                        for attrib_key in pred[key].keys():
                            pred[key][attrib_key]["scores"] = pred[key][attrib_key]["scores"][:i]
                            pred[key][attrib_key]["labels"] = pred[key][attrib_key]["labels"][:i]
                    else:
                        pred[key] = pred[key][:i]
                break 
                
        pred_boxes = pred["boxes"].long()
        if show_attrib:

            pred_attrib_labels = [dataset.reverse_attrib_mappings[target_attrib][int(x)] for x in pred['attributes'][target_attrib]["labels"]]
            pred_attrib_scores = pred['attributes'][target_attrib]["scores"]
            pred_labels = [f"{score:.3f}" + "\n" + f"{attrib_label}: {attrib_score:.3f}"
                            for score, attrib_label, attrib_score in zip(pred['scores'], pred_attrib_labels, pred_attrib_scores)]
            
        else:
            # label=1 is building, not sure I need to make that look nicer
            pred_labels = [f"{score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
        # print(pred["labels"])
        # print(pred["scores"])
        output_image = draw_bounding_boxes(output_image, pred_boxes, pred_labels, colors="red", font='Roboto-Regular.ttf', font_size=32)
        
    
    plt.figure(figsize=(24, 24))
    plt.imshow(output_image.permute(1, 2, 0))
    if save:
        if subfolder is None:
            subfolder = 'eval_image'
        if target_attrib:
            filepath = join(subfolder, f"epoch_{epoch}_image_{image_id}_{target_attrib}.jpg")
        else:
            filepath = join(subfolder, f"epoch_{epoch}_image_{image_id}.jpg")
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)  # Specify the path and filename
        print(f"Saving image as {filepath}...")
    else:
        plt.show()
    plt.close() 

# def custom_fasterrcnn_resnet50_fpn(
#     pretrained=False, progress=True, num_classes=91, attrib_mappings=None, attribute_weights_dict=None, pretrained_backbone=True, trainable_backbone_layers=None, num_heads = 2,
#     attention_per_attrib=True, custom_box_predictor=False, use_reduced_features_for_attrib=False, use_attention=True, parallel_backbone=False, **kwargs
# ):
#     """
#     Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

#     Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
#     Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.

#     The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
#     image, and should be in ``0-1`` range. Different images can have different sizes.

#     The behavior of the model changes depending if it is in training or evaluation mode.

#     During training, the model expects both the input tensors, as well as a targets (list of dictionary),
#     containing:

#         - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
#           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
#         - labels (``Int64Tensor[N]``): the class label for each ground-truth box

#     The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
#     losses for both the RPN and the R-CNN.

#     During inference, the model requires only the input tensors, and returns the post-processed
#     predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
#     follows, where ``N`` is the number of detections:

#         - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
#           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
#         - labels (``Int64Tensor[N]``): the predicted labels for each detection
#         - scores (``Tensor[N]``): the scores of each detection

#     For more details on the output, you may refer to :ref:`instance_seg_output`.

#     Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

#     Example::

#         >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         >>> # For training
#         >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
#         >>> boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
#         >>> labels = torch.randint(1, 91, (4, 11))
#         >>> images = list(image for image in images)
#         >>> targets = []
#         >>> for i in range(len(images)):
#         >>>     d = {}
#         >>>     d['boxes'] = boxes[i]
#         >>>     d['labels'] = labels[i]
#         >>>     targets.append(d)
#         >>> output = model(images, targets)
#         >>> # For inference
#         >>> model.eval()
#         >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#         >>> predictions = model(x)
#         >>>
#         >>> # optionally, if you want to export the model to ONNX:
#         >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on COCO train2017
#         progress (bool): If True, displays a progress bar of the download to stderr
#         num_classes (int): number of output classes of the model (including the background)
#         pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
#         trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
#             Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
#             passed (the default) this value is set to 3.
#     """
#     trainable_backbone_layers = _validate_trainable_layers(
#         pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
#     )

#     if pretrained:
#         # no need to download the backbone if pretrained is set
#         pretrained_backbone = False

#     if attrib_mappings == None:
#         raise ValueError("need to map attributes")

#     if pretrained_backbone:
#         backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=progress,
#                             norm_layer=misc_nn_ops.FrozenBatchNorm2d)
#         if parallel_backbone:
#             parallel_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=progress,
#                                          norm_layer=misc_nn_ops.FrozenBatchNorm2d)
#     else:
#         backbone = resnet50(progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
#         if parallel_backbone:
#             parallel_backbone = resnet50(progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)

#     backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
#     if parallel_backbone:
#         parallel_backbone = _resnet_fpn_extractor(parallel_backbone, trainable_backbone_layers)
#     model = CustomFasterRCNN(backbone, num_classes=num_classes, attribute_mappings=attrib_mappings, attribute_weights_dict=attribute_weights_dict,
#         attention_per_attrib=attention_per_attrib, custom_box_predictor=custom_box_predictor, use_attention=use_attention, num_heads= num_heads,
#         use_reduced_features_for_attrib=use_reduced_features_for_attrib, parallel_backbone=parallel_backbone, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
#         model.load_state_dict(state_dict)
#         overwrite_eps(model, 0.0)
#     return model


# def custom_fasterrcnn_resnet101_fpn(
#     pretrained=False, progress=True, num_classes=91, attrib_mappings=None, attribute_weights_dict=None, pretrained_backbone=True, trainable_backbone_layers=None,
#     attention_per_attrib=True, custom_box_predictor=False, num_heads = 2, parallel_backbone=None, **kwargs
# ):
#     """
#     Constructs a Faster R-CNN model with a ResNet-101-FPN backbone.

#     Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
#     Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.

#     The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
#     image, and should be in ``0-1`` range. Different images can have different sizes.

#     The behavior of the model changes depending if it is in training or evaluation mode.

#     During training, the model expects both the input tensors, as well as a targets (list of dictionary),
#     containing:

#         - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
#           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
#         - labels (``Int64Tensor[N]``): the class label for each ground-truth box

#     The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
#     losses for both the RPN and the R-CNN.

#     During inference, the model requires only the input tensors, and returns the post-processed
#     predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
#     follows, where ``N`` is the number of detections:

#         - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
#           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
#         - labels (``Int64Tensor[N]``): the predicted labels for each detection
#         - scores (``Tensor[N]``): the scores of each detection

#     For more details on the output, you may refer to :ref:`instance_seg_output`.

#     Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

#     Example::

#         >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         >>> # For training
#         >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
#         >>> boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
#         >>> labels = torch.randint(1, 91, (4, 11))
#         >>> images = list(image for image in images)
#         >>> targets = []
#         >>> for i in range(len(images)):
#         >>>     d = {}
#         >>>     d['boxes'] = boxes[i]
#         >>>     d['labels'] = labels[i]
#         >>>     targets.append(d)
#         >>> output = model(images, targets)
#         >>> # For inference
#         >>> model.eval()
#         >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#         >>> predictions = model(x)
#         >>>
#         >>> # optionally, if you want to export the model to ONNX:
#         >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on COCO train2017
#         progress (bool): If True, displays a progress bar of the download to stderr
#         num_classes (int): number of output classes of the model (including the background)
#         pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
#         trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
#             Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
#             passed (the default) this value is set to 3.
#     """

#     trainable_backbone_layers = _validate_trainable_layers(
#         pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
#     )

#     if pretrained:
#         # no need to download the backbone if pretrained is set
#         pretrained_backbone = False

#     if attrib_mappings == None:
#         raise ValueError("need to map attributes")

#     if pretrained_backbone:
#         backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=progress,
#                             norm_layer=misc_nn_ops.FrozenBatchNorm2d)
#         if parallel_backbone:
#             parallel_backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=progress,
#                                   norm_layer=misc_nn_ops.FrozenBatchNorm2d)
#     else:
#         backbone = resnet101(progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)

#         if parallel_backbone:
#             parallel_backbone = resnet101(progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)

#     backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
#     if parallel_backbone:
#         parallel_backbone = _resnet_fpn_extractor(parallel_backbone, trainable_backbone_layers) # should this use a different trainable_backbone_layers?
#     model = CustomFasterRCNN(backbone, num_classes=num_classes, attribute_mappings=attrib_mappings, attribute_weights_dict=attribute_weights_dict,
#         attention_per_attrib=attention_per_attrib, custom_box_predictor=custom_box_predictor, num_heads=num_heads, parallel_backbone=parallel_backbone, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
#         model.load_state_dict(state_dict)
#         overwrite_eps(model, 0.0)
#     return model