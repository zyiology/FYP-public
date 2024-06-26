### CUSTOM DATASET FOR modified faster rcnn
# importantly implements attributes for each bounding box

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
import json
from torchvision.transforms.v2 import functional as F
import os

class CustomFRCNNAttentionDataset(torch.utils.data.Dataset):
    '''
    Class defining a custom dataset for use with the modified faster rcnn model.
    Importantly, defines attributes for each bounding box.
    Args:
        root (str): root directory of dataset
        transforms (torchvision.transforms): transforms to apply to images and targets
        annotations_path (str): path to COCO annotations file
        attrib_mappings (dict[dict[str, int]]): dictionary of attribute classes to their integer values
        classes (list[str]): list of class names
        exclude (list[int]): list of image ids to exclude
        offset (int): offset to add to image ids
        ignore_occluded (bool): whether to ignore occluded annotations (each annotation has a default attribute 'occluded')
    '''

    def __init__(self, root, transforms, annotations_path, attrib_mappings=None, classes=None, exclude=None, offset=0, ignore_occluded=True):
        if exclude is None:
            exclude = []
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.transforms = transforms
        
        with open(annotations_path) as json_file:
            self.coco_annotations = json.load(json_file)

        # Filter images with bounding boxes
        # if at least one annotation has a bounding box and matches the id, then it will be included
        # if ignore_occluded is True, then only annotations with occluded=False will be included
        if ignore_occluded:
            self.images = [img for img in self.coco_annotations['images'] 
                        if img['id'] not in exclude and 
                        any(ann['image_id'] == img['id'] and 'bbox' in ann and ann['attributes']['occluded'] is not True
                            for ann in self.coco_annotations['annotations'])] 
        else:            
            self.images = [img for img in self.coco_annotations['images'] 
                        if img['id'] not in exclude and 
                        any(ann['image_id'] == img['id'] and 'bbox' in ann
                            for ann in self.coco_annotations['annotations'])] 

        self.ignore_occluded = ignore_occluded

        # load class names from annotations file if not provided
        if classes:
            self.classes = classes
        else:
            self.classes = [category['name'] for category in self.coco_annotations['categories']]
            
        # if no background class, add it
        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        # attrib_mappings stores the mapping of attribute names to their integer values, and reverse_ is the inverse mapping
        self.attrib_mappings = attrib_mappings
        self.reverse_attrib_mappings = {}
        for attrib_name, mapping_dict in attrib_mappings.items():
            self.reverse_attrib_mappings[attrib_name] = {v:k for k,v in mapping_dict.items()}
        

    def __getitem__(self, idx):
        img_id = self.images[idx]['id']
        img_path = os.path.join(self.img_dir, self.images[idx]['file_name'])
        
        # load images and masks       
        img = read_image(img_path, mode=ImageReadMode.RGB) # ignore alpha channel if present, only read RGB

         # Get annotations for this image
        annotations = [ann for ann in self.coco_annotations['annotations'] if ann['image_id'] == img_id]

        if self.ignore_occluded:
            annotations = [ann for ann in annotations if ann['attributes']['occluded'] is False]

        # Extract bounding boxes, labels, etc.
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        area = [ann['area'] for ann in annotations]

        attributes = {}

        try:
            for key, mapping_dict in self.attrib_mappings.items():
                attributes[key] = torch.as_tensor([mapping_dict[ann['attributes'][key]] for ann in annotations], dtype=torch.int64)
        # catch exception if key not found
        except KeyError as e:
            print(f"KeyError: {e}")
            for ann in annotations:
                print(ann['attributes'].keys())
            print(img_id)
            raise KeyError

        num_objs = len(labels)
        
        # suppose all instances are not crowd - required variable for COCO evaluation
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        # convert boxes to right format
        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = img_id
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = iscrowd
        target["attributes"] = attributes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # same as __getitem__ but without the image to save memory and speed up processing
    def get_target(self, idx):
        img_id = self.images[idx]['id']
        img_path = os.path.join(self.img_dir, self.images[idx]['file_name'])
        img_size = Image.open(img_path).size

         # Get annotations for this image
        annotations = [ann for ann in self.coco_annotations['annotations'] if ann['image_id'] == img_id]

        # Extract bounding boxes, labels, etc.
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        area = [ann['area'] for ann in annotations]

        attributes = {}

        try:
            for key, mapping_dict in self.attrib_mappings.items():
                attributes[key] = torch.as_tensor([mapping_dict[ann['attributes'][key]] for ann in annotations], dtype=torch.int64)
        except KeyError as e:
            print(f"KeyError: {e}")
            for ann in annotations:
                print(ann['attributes'].keys())
            print(img_id)
            raise KeyError

        num_objs = len(labels)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # convert boxes to right format
        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=img_size)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = img_id
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = iscrowd
        target["attributes"] = attributes

        if self.transforms is not None:
            target = self.transforms(target)

        return target

    def __len__(self):
        return len(self.images)