from collections import Counter
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .base_data import BaseDataset


class DetectionDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def category_id_to_name(self, category_id: int):
        """ Returns a human interpretable name corresponding to a category id """
        return self.ds[self.split].features['annotations'].feature['category']['name'].names[category_id]
    
    @staticmethod
    def visualize_image(
        image: Union[torch.Tensor, Image.Image],
        bboxs: list[tuple[int, int, int, int]],
        labels: Optional[list[str]] = None,
        return_image: bool = False
    ) -> Optional[Image.Image]:
        # normalize to numpy array
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        # if they just want to get back an image-with-boxes
        if return_image:
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            font = ImageFont.load_default()
            for i, bbox in enumerate(bboxs):
                x, y, w, h = map(int, bbox)
                draw.rectangle([x, y, x+w, y+h], outline="red", width=1)
                if labels:
                    draw.text((x, y), str(labels[i]), font=font, fill="white")
            return pil

        # otherwise show via matplotlib as before
        fig, ax = plt.subplots()
        ax.imshow(image)
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox
            ax.add_patch(plt.Rectangle((x,y), w, h, fill=False, edgecolor='red', linewidth=1))
            if labels:
                ax.text(x, y, labels[i], color='white', backgroundcolor='black', fontsize=8)
        ax.axis('off')
        plt.show()

    
    def __getitem__(self, index: int):
        """ Returns image, human-interpretable label, category id, and bounding boxes  """
        labels = [self.category_id_to_name(label['name']) for label in self.ds[self.split][index]['annotations']['category']]
        bbox = np.array(self.ds[self.split][index]['annotations']['bbox'])
        image = self.ds[self.split][index]['image']

        assert isinstance(image, Image.Image), "Image must be a PIL.Image.Image before applying transforms"
        original_w, original_h = image.size

        if self.tf:
            image = self.tf(image)

        if isinstance(image, Image.Image):
            new_w, new_h = image.size
        elif isinstance(image, torch.Tensor):
            print(image.shape)
            new_h, new_w = image.shape[1], image.shape[2]
        else:
            raise ValueError(f"Image must be a PIL.Image.Image or a torch.Tensor. Got {type(image)}")

        scale_x = new_w / original_w
        scale_y = new_h / original_h
        bbox[:, 0] *= scale_x
        bbox[:, 1] *= scale_y
        bbox[:, 2] *= scale_x
        bbox[:, 3] *= scale_y
        bbox = bbox.astype(int)
        
        if self.visualize:
            self.visualize_image(image, bbox, labels)
        return {'pixel_values': image, 'labels': labels, 'bbox': bbox, 'unique_labels': Counter(labels), 'total_labels' : len(labels)}