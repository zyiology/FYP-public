import torch
from PIL import Image
from torchvision import tv_tensors
# import fiftyone.utils.coco as fouc
from torchvision.io import read_image, ImageReadMode
import json
from torchvision.transforms.v2 import functional as F
import os

class CustomFRCNNAttentionDataset(torch.utils.data.Dataset):
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

        if classes:
            self.classes = classes
        else:
            self.classes = [category['name'] for category in self.coco_annotations['categories']]
            
        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.attrib_mappings = attrib_mappings
        self.reverse_attrib_mappings = {}


        for attrib_name, mapping_dict in attrib_mappings.items():
            self.reverse_attrib_mappings[attrib_name] = {v:k for k,v in mapping_dict.items()}
        

    def __getitem__(self, idx):
        img_id = self.images[idx]['id']
        img_path = os.path.join(self.img_dir, self.images[idx]['file_name'])
        
        # load images and masks       
        img = read_image(img_path, mode=ImageReadMode.RGB)#_ALPHA if img_path[-4:] == ".png" else ImageReadMode.RGB)

         # Get annotations for this image
        annotations = [ann for ann in self.coco_annotations['annotations'] if ann['image_id'] == img_id]

        if self.ignore_occluded:
            annotations = [ann for ann in annotations if ann['attributes']['occluded'] is False]

        # Extract bounding boxes, labels, etc.
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        area = [ann['area'] for ann in annotations]

        attributes = {}
        #make this into a list later

        try:
            for key, mapping_dict in self.attrib_mappings.items():
                attributes[key] = torch.as_tensor([mapping_dict[ann['attributes'][key]] for ann in annotations], dtype=torch.int64)
        except KeyError as e:
            print(f"KeyError: {e}")
            for ann in annotations:
                print(ann['attributes'].keys())
            print(img_id)
            raise KeyError

        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        num_objs = len(labels)
        
        # suppose all instances are not crowd
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
        #make this into a list later

        try:
            for key, mapping_dict in self.attrib_mappings.items():
                attributes[key] = torch.as_tensor([mapping_dict[ann['attributes'][key]] for ann in annotations], dtype=torch.int64)
        except KeyError as e:
            print(f"KeyError: {e}")
            for ann in annotations:
                print(ann['attributes'].keys())
            print(img_id)
            raise KeyError

        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

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