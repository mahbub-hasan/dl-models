import timeit
from PIL import Image

import torch
from torchvision import transforms
from project_configuration import ProjectConfiguration
from model import BuildModel

if __name__ == '__main__':
    config = ProjectConfiguration(image_size=224, image_channel=1,patch_size=16,attention_head=4, attention_layer=12,
                                  batch_size=1)
    img_transform = transforms.Compose([
        transforms.Resize(size=(config.image_size, config.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # image_path = '/Users/mahbubhasan/Documents/Research_Self/Melanoma_Seg/dataset/training/images/ISIC_0024464.jpg'

    # img = Image.open(image_path).convert('RGB')

    # img = img_transform(img)

    img = torch.randn(config.batch_size, config.image_channel, config.image_size, config.image_size)
    model = BuildModel(config=config)
    start_time = timeit.default_timer()

    output = model(img)

    end_time = timeit.default_timer()
    print(f"Time diff: {end_time - start_time:.3f}s")
    print(output.shape)
