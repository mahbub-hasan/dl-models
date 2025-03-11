import torch
import torch.nn as nn
from project_config import ProjectConfig
from torchvision import datasets, transforms


class DataPreprocess(nn.Module):
    def __init__(self, config: ProjectConfig, train_data_path: str, test_data_path: str):
        super(DataPreprocess, self).__init__()
        self.config = config
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.image_transforms = transforms.Compose([
            transforms.Resize(size=(config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])

    def forward(self):
        train_data = datasets.ImageFolder(root=self.train_data_path, transform=self.image_transforms)
        test_data = datasets.ImageFolder(root=self.test_data_path, transform=self.image_transforms)

        return train_data, test_data, len(train_data.class_to_idx), train_data.class_to_idx
