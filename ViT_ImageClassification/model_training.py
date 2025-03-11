import sys

import torch
import torch.nn as nn

from project_config import ProjectConfig
from tqdm import tqdm

sys.path.append('ViT')
from ViT.vit_model import ViTModel

sys.path.append('DataPrepare')
from DataPrepare.make_dataset import MakeDataset

if __name__ == '__main__':
    # setup hyperparameter for this project
    config = ProjectConfig(image_size=224,
                           image_channel=3,
                           image_patch_size=16,
                           attention_head=8,
                           attention_layer=12,
                           hidden_layer=2048,
                           eps=1e-6,
                           dropout=0.1,
                           batch_size=4,
                           epochs=50)

    # seed define
    torch.manual_seed(42)

    # define train and test data path
    train_data_path = '/Users/mahbubhasan/Documents/Research_Self/PyTorch/ClassificationProblem/data/train_data'
    test_data_path = '/Users/mahbubhasan/Documents/Research_Self/PyTorch/ClassificationProblem/data/test_data'

    # make dataset
    make_dataset = MakeDataset(config=config, val_split=0.1, train_data_path=train_data_path,
                               test_data_path=test_data_path)

    train_dataset, val_dataset, test_dataset, num_of_class, class_index_map = make_dataset()

    # Build Model
    model = ViTModel(config=config, number_of_class=num_of_class)

    # define loss and optimizer for this project
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # define training loop
    for epoch in range(config.epochs):
        model.train()

        # print("Start Training Loop")
        # define some arrays for training
        train_label = []
        train_predict_label = []
        train_running_loss = 0.0

        for batch_size, (image, label) in enumerate(tqdm(train_dataset, position=0)):
            img = image
            labels = label.float().requires_grad_(False)

            y_pred, attention_scores = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1).float().requires_grad_(True)

            train_label.extend(labels)
            train_predict_label.extend(y_pred_label)

            print(f"\nTraining Phase.......")
            print(f"Batch: {batch_size}")
            print(f"{y_pred_label}")
            print(f"{labels}\n")

            loss = loss_fn(y_pred_label, labels)

            # start backpropagation
            # print("Start Training Backpropagation")
            model.zero_grad()
            loss.backward()
            optim.step()

            # now added all the loss in the array
            # print("Adding training losses")
            train_running_loss += loss.item()

        # print("calculate whole training loss")
        train_loss = train_running_loss / (len(train_dataset))

        # start val loop
        # print("Start model evaluation")
        model.eval()
        val_labels = []
        val_predict_label = []
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_size, (image, label) in enumerate(tqdm(val_dataset, position=0)):
                img = image
                labels = label.float().requires_grad_(False)

                y_pred, attention_scores = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1).float().requires_grad_(True)

                print(f"\nValidation Phase.......")
                print(f"Batch: {batch_size}")
                print(f"{y_pred_label}")
                print(f"{labels}\n")

                val_labels.extend(labels)
                val_predict_label.extend(y_pred_label)

                loss = loss_fn(y_pred_label, labels)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_dataset)

        # output
        print("-"*30)
        print(f"Epoch: {epoch+1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Training Accuracy: {sum(1 for x, y in zip(train_predict_label, train_label) if x==y)/len(train_label):.4f}")
        print(f"Validation Accuracy: {sum(1 for x, y in zip(val_predict_label, val_labels) if x==y)/len(val_labels):.4f}")
        print("-"*30)

