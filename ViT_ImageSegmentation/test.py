import timeit

import torch
from project_configuration import ProjectConfiguration
from encoder.image_encoder import ImageEncoder
from decoder.image_decoder import ImageDecoder

if __name__ == '__main__':
    config = ProjectConfiguration()
    img = torch.randn(size=(config.batch_size, config.image_channel, config.image_size, config.image_size))

    start_time = timeit.default_timer()

    encoded_image = ImageEncoder(config=config)
    decoder_image = ImageDecoder(config=config)
    encode = encoded_image(img)
    decode = decoder_image(encode)

    end_time = timeit.default_timer()
    print(f"Time diff: {end_time - start_time:.3f}s")
    print(decode.shape)
