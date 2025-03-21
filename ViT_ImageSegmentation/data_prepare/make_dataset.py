import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from project_configuration import ProjectConfiguration


class DataPrepared(Dataset):
    def __init__(self, config: ProjectConfiguration, root_dir, data_length: int, transform=None):
        super(DataPrepared, self).__init__()
        self.config = config
        self.root_dir = root_dir
        self.transform = transform
        self.data_length = data_length
        self.img_transform = transforms.Compose([
            transforms.Resize(size=(config.image_size, config.image_size)),
            transforms.Grayscale(num_output_channels=config.image_channel),
            transforms.ToTensor()
        ])
        self.images = sorted([f for f in os.listdir(os.path.join(root_dir, 'images')) if not f.startswith(".")])[:data_length]
        self.mask = sorted([f for f in os.listdir(os.path.join(root_dir, 'mask')) if not f.startswith(".")])[:data_length]

    def __len__(self):
        assert len(self.images) == len(self.mask)
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        mask_image_name = self.mask[index]

        image_path = os.path.join(self.root_dir, 'images', image_name)
        mask_image_path = os.path.join(self.root_dir, 'mask', mask_image_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_image_path).convert('L')

        if self.transform:
            image = self.img_transform(image)
            mask = self.img_transform(mask)

        return {"source": image, "target": mask, "index": index}


class MakeDataset(nn.Module):
    def __init__(self, config: ProjectConfiguration, root_dir: str, test_split_ratio: float, data_length: int,
                 transform=None):
        super(MakeDataset, self).__init__()
        self.config = config
        self.test_split_ratio = test_split_ratio
        self.data = DataPrepared(config, root_dir, data_length, transform)
        self.generator = torch.Generator().manual_seed(42)

    def forward(self):
        train_data, val_data = random_split(dataset=self.data, lengths=[1-self.test_split_ratio, self.test_split_ratio],
                                            generator=self.generator)
        val_data, test_data = random_split(dataset=val_data, lengths=[1-self.test_split_ratio, self.test_split_ratio],
                                           generator=self.generator)

        # make DataLoader for training image
        train_dataset = DataLoader(dataset=train_data, batch_size=self.config.batch_size, shuffle=True)

        # make DataLoader for val image
        val_dataset = DataLoader(dataset=val_data, batch_size=self.config.batch_size, shuffle=True)

        # make DataLoader for test image
        test_dataset = DataLoader(dataset=test_data, batch_size=self.config.batch_size, shuffle=False)

        return train_dataset, val_dataset, test_dataset
