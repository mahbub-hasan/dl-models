import os

from torch.utils.data import Dataset
from torchvision import transforms
from project_configuration import ProjectConfiguration
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, config: ProjectConfiguration, data_path: str, image_transforms=None):
        self.config = config
        self.data_path = data_path
        self.images = sorted(os.listdir(os.path.join(data_path, 'images')))[:config.dataset_size]
        self.masks = sorted(os.listdir(os.path.join(data_path, 'mask')))[:config.dataset_size]

        if image_transforms:
            self.image_transforms = transforms.Compose([
                transforms.Resize(size=(config.image_size, config.image_size)),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')

        return self.image_transforms(image), self.image_transforms(mask)
