from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, split, transform, visualize, **kwargs):
        super().__init__()
        self.split: str = split
        self.tf: transforms = transform
        self.visualize: bool = visualize
        self.ds: Dataset = load_dataset(**kwargs)

    def __len__(self):
        return len(self.ds[self.split])

    def __iter__(self):
        return iter(self.ds[self.split])

    def __next__(self):
        return next(self.ds[self.split])

    @staticmethod
    def visualize_image(
        image: Union[torch.Tensor, Image.Image],
        bboxs: list[tuple[int, int, int, int]],
        labels: Optional[list[str]] = None,
        return_image: bool = False,
    ) -> Optional[Image.Image]:
        """
        Visualize a list of bounding boxes on an image.
        """
        # normalize to numpy array
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)

        if return_image:
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            font = ImageFont.load_default()
            for i, bbox in enumerate(bboxs):
                x, y, w, h = map(int, bbox)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
                if labels:
                    draw.text((x, y), str(labels[i]), font=font, fill="white")
            return pil

        # otherwise show via matplotlib as before
        fig, ax = plt.subplots()
        ax.imshow(image)
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox
            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1)
            )
            if labels:
                ax.text(
                    x, y, labels[i], color="white", backgroundcolor="black", fontsize=8
                )
        ax.axis("off")
        plt.show()
