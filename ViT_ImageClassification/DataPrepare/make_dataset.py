import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from project_config import ProjectConfig
from load_data import DataPreprocess


class MakeDataset(nn.Module):
    def __init__(self, config: ProjectConfig, val_split: float, train_data_path: str, test_data_path: str):
        super(MakeDataset, self).__init__()
        self.config = config
        self.val_split = val_split
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data_prepare = DataPreprocess(config=config, train_data_path=train_data_path,
                                           test_data_path=test_data_path)

    def forward(self):
        train_data, test_data, num_of_class, class_index_map = self.data_prepare()
        # split train_data into train and val data
        _split_train_data, _split_val_data = random_split(dataset=train_data, lengths=[16, 4])

        # make train_dataset
        train_dataset = DataLoader(dataset=_split_train_data, batch_size=self.config.batch_size, shuffle=True)

        # make val_dataset
        val_dataset = DataLoader(dataset=_split_val_data, batch_size=self.config.batch_size, shuffle=True)

        # make test_dataset
        test_dataset = DataLoader(dataset=test_data, batch_size=self.config.batch_size, shuffle=False)

        return train_dataset, val_dataset, test_dataset, num_of_class, class_index_map
