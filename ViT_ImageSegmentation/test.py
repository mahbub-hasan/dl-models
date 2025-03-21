import timeit

import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration
from data_prepare.make_dataset import MakeDataset
from model_build.model import BuildModel
from model_build.dice_accuracy import DiceAccuracy
from model_build.train_loop import TrainLoop
from torchinfo import summary

if __name__ == '__main__':
    # define project configuration
    config = ProjectConfiguration(image_size=224,
                                  image_channel=1,
                                  patch_size=16,
                                  attention_head=4,
                                  attention_layer=12,
                                  batch_size=8,
                                  eps=1e-7,
                                  hidden_layer=2048,
                                  dropout=0.1,
                                  learning_rate=1e-4,
                                  epochs=10)

    # define data loader
    train_dataset, val_dataset, test_dataset = MakeDataset(config=config,
                                                           root_dir='/Users/mahbubhasan/Documents/Research_Self/Melanoma_Seg/dataset/training',
                                                           data_length=2000,  # image count from data
                                                           test_split_ratio=0.2, transform=True).forward()

    # print train_dataset
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # now it's time to build the model
    model = BuildModel(config=config)

    # print model summary
    summary(model, input_size=(config.batch_size,config.image_channel, config.image_size, config.image_size))

    # let's define loss and optimizer for the model
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dice_loss = DiceAccuracy(config=config)

    # let's run the train loop
    t_loop = TrainLoop(config=config,
                       model=model,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       dice_loss_fn=dice_loss)
    train_losses, train_dice_losses = t_loop(train_dataset)


    # # image_path = '/Users/mahbubhasan/Documents/Research_Self/Melanoma_Seg/dataset/training/images/ISIC_0024464.jpg'
    #
    # # img = Image.open(image_path).convert('RGB')
    #
    # # img = img_transform(img)
    #
    # img = torch.randn(config.batch_size, config.image_channel, config.image_size, config.image_size)
    # model = BuildModel(config=config)
    # start_time = timeit.default_timer()
    #
    # output = model(img)
    #
    # end_time = timeit.default_timer()
    # print(f"Time diff: {end_time - start_time:.3f}s")
    # print(output.shape)
