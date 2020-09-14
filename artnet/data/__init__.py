import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.data
from PIL import Image
from torchvision.datasets import VisionDataset, VOCSegmentation, VOCDetection


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
        # self.imgs = list(sorted((Path(self.root) / "JPEGImages").iterdir()))
        # self.annotations = list(sorted((Path(self.root) / "Annotations").iterdir()))
        self.imgs = list((self._root / 'JPEGImages').glob('**/*.jpg'))

    def __getitem__(self, idx):
        # Handle slices
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, tuple):
            raise NotImplementedError('Tuple as index')

        # load images and bounding boxes
        img_path = self.imgs[idx]
        rel_path = img_path.relative_to(self._root / 'JPEGImages')

        img = Image.open(img_path).convert("RGB")
        boxes = []

        try:
            annotation_path = next((self._root / 'Annotations').glob(str(rel_path.with_suffix('.*'))))
            # annotation_path = self.annotations[idx]
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
        except (StopIteration, FileNotFoundError) as e:  # There's no file there.
            num_objs = 0
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

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
            # print('Transforms: ', self.transforms)
            # print('Transforms list: ', self.transforms.transforms)
            # print('img: ', img)
            # print('target: ', target)
            img, target = self.transforms(img, target)

        return img, target  # , image_id

    def __len__(self):
        return len(self.imgs)

    @property
    def _root(self):
        return Path(self.root)

class VOCDetectionSubset(VOCDetection):

    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None,
                 transforms=None, names=[]):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)

        self.names = names

    def __getitem__(self, index):
        img, annotation = super().__getitem__(index)
        annotation['annotation']['object'] = self.filter_annotation_by_object_names(annotation)
        return img, annotation

    def filter_annotation_by_object_names(self, annotation):
        return list(filter(lambda o: o['name'] in self.names, annotation['annotation']['object']))