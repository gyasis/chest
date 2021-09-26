import albumentations as A
from torch.utils.data import Dataset, DataLoader
from collection import OrderedDict

class ChestXRayDataset(Dataset):
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms
        
        
    def __len__(self):
        return(len(self.images))
    
    