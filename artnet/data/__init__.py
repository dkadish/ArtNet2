import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.data
from PIL import Image

from torchvision.datasets import VisionDataset


class ArtNetDataset(VisionDataset):

    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, transforms=transforms,
                                    target_transform=target_transform)

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted((Path(self.root) / "JPEGImages").iterdir()))
        self.annotations = list(sorted((Path(self.root) / "Annotations").iterdir()))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        annotation_path = self.annotations[idx]
        img = Image.open(img_path).convert("RGB")

        boxes = []
        tree = ET.parse(Path(annotation_path))
        tree.getroot()
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bbox = (xmin, ymin, xmax, ymax)
            boxes.append(bbox)

        # get bounding box coordinates for each mask
        num_objs = len(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            print('Transforms: ', self.transforms)
            print('Transforms list: ', self.transforms.transforms)
            print('img: ', img)
            print('target: ', target)
            img, target = self.transforms(img, target)

        return img, target#, image_id

    def __len__(self):
        return len(self.imgs)
