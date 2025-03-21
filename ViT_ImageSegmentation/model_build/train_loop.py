import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration
from tqdm import tqdm


class TrainLoop(nn.Module):
    def __init__(self, config: ProjectConfiguration, model: nn.Module,
                 loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                 dice_loss_fn, device: torch.device = None):
        super(TrainLoop, self).__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dice_loss_fn = dice_loss_fn
        self.device = device

    def forward(self, data: torch.utils.data.DataLoader):
        self.model.eval()
        train_losses = []
        train_dice_losses = []
        train_loss = 0.0
        train_dice_loss = 0.0
        for epoch in range(self.config.epochs):
            self.model.train()
            train_running_loss = 0.0
            train_running_dice_loss = 0.0
            for idx, img_mask in enumerate(tqdm(iterable=data, position=0, leave=True)):
                image = img_mask["source"].float()
                mask = img_mask['target'].float()

                y_pred = self.model(image)
                self.optimizer.zero_grad()

                loss = self.loss_fn(y_pred, mask)
                dice_loss = self.dice_loss_fn(y_pred, mask)

                train_running_loss += loss.item()
                train_running_dice_loss += dice_loss.item()

                loss.backward()
                self.optimizer.step()

            train_loss = train_running_loss / (idx + 1)
            train_dice_loss = train_running_dice_loss / (idx + 1)

            print(f"Epoch: {epoch + 1}")
            print("-"*30)
            print(f"Train Loss: {train_loss:.2f}")
            print(f"Train Dice Loss: {train_dice_loss:.2f}")
            print("-"*30)

            train_losses.append(train_loss)
            train_dice_losses.append(train_dice_loss)

        # saving the model
        torch.save(self.model.state_dict(), 'model_01.pth')

        return train_losses, train_dice_losses
