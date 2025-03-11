import torch
from project_configuration import ProjectConfiguration
from encoder.image_econder import ImageEncoder
if __name__ == '__main__':
    config = ProjectConfiguration()
    img = torch.randn(size=(config.batch_size, config.image_channel, config.image_size, config.image_size))
    encoded_image = ImageEncoder(config=config)
    out = encoded_image(img)
    print(out.shape)
