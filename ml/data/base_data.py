from datasets import load_dataset
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
