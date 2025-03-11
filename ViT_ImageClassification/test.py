import sys
import torch
from project_config import ProjectConfig

sys.path.append('ViT')
from ViT.vit_model import ViTModel

sys.path.append('DataPrepare')
from DataPrepare.make_dataset import MakeDataset

if __name__ == '__main__':
    config = ProjectConfig(image_size=224, image_channel=3, image_patch_size=16, attention_head=8, attention_layer=12,
                           hidden_layer=2048, eps=1e-6, dropout=0.1, batch_size=4, epochs=1)
    # img = torch.randn(config.batch_size, config.image_channel, config.image_size, config.image_size)

    train_data_path = '/Users/mahbubhasan/Documents/Research_Self/PyTorch/ClassificationProblem/data/train_data'
    test_data_path = '/Users/mahbubhasan/Documents/Research_Self/PyTorch/ClassificationProblem/data/test_data'

    dataset = MakeDataset(config=config, val_split=0.1, train_data_path=train_data_path,
                          test_data_path=test_data_path)

    train_dataset, val_dataset, test_dataset, num_of_class, class_index_map = dataset()

    print(len(train_dataset))

    print(class_index_map)
    print(num_of_class)

    print("From Train Dataset")
    for batch, (image, label) in enumerate(train_dataset):
        print(f"Batch: {batch}, Label: {label}")

    print("From Val Dataset")
    for batch, (image, label) in enumerate(val_dataset):
        print(f"Batch: {batch}, Label: {label}")

    # model = ViTModel(config=config, number_of_class=2)
    # x, _ = model(img)
    #
    # print(x)
