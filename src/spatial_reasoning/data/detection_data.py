from collections import Counter

import numpy as np
import torch
from PIL import Image

from .base_data import BaseDataset


class CoCoDetectionDataset(BaseDataset):
    def __init__(self, **kwargs):
        """
        NOTE: Inherit this class method to adapt to your own dataset.
        This dataset is designed to work with the COCO dataset.
        To use it, run:
        ```
        ds = CoCoDetectionDataset(
            path="shunk031/MSCOCO",
            year=2014,
            coco_task="instances",
            decode_rle=True,
            split='validation',
            visualize=False,
            trust_remote_code=True
        )
        print(ds[0])
        ```
        """
        super().__init__(**kwargs)

    def category_id_to_name(self, category_id: int):
        """Returns a human interpretable name corresponding to a category id"""
        return (
            self.ds[self.split]
            .features["annotations"]
            .feature["category"]["name"]
            .names[category_id]
        )

    def __getitem__(self, index: int):
        """
        Returns image, human-interpretable label, category id, and bounding boxes
        """
        labels = [
            self.category_id_to_name(label["name"])
            for label in self.ds[self.split][index]["annotations"]["category"]
        ]
        bbox = np.array(self.ds[self.split][index]["annotations"]["bbox"])
        image = self.ds[self.split][index]["image"]

        assert isinstance(image, Image.Image), (
            "Image must be a PIL.Image.Image before applying transforms"
        )
        original_w, original_h = image.size

        if self.tf:
            image = self.tf(image)

        if isinstance(image, Image.Image):
            new_w, new_h = image.size
        elif isinstance(image, torch.Tensor):
            print(image.shape)
            new_h, new_w = image.shape[1], image.shape[2]
        else:
            raise ValueError(
                f"Image must be a PIL.Image.Image or a torch.Tensor. Got {type(image)}"
            )

        scale_x = new_w / original_w
        scale_y = new_h / original_h
        bbox[:, 0] *= scale_x
        bbox[:, 1] *= scale_y
        bbox[:, 2] *= scale_x
        bbox[:, 3] *= scale_y
        bbox = bbox.astype(int)

        if self.visualize:
            self.visualize_image(image, bbox, labels)
        return {
            "pixel_values": image,
            "labels": labels,
            "bbox": bbox,
            "unique_labels": Counter(labels),
            "total_labels": len(labels),
        }
