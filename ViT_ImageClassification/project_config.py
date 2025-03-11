class ProjectConfig:
    def __init__(self,
                 image_size: int,
                 image_channel: int,
                 image_patch_size: int,
                 attention_head: int,
                 attention_layer: int,
                 hidden_layer: int,
                 eps: float,
                 dropout: float,
                 batch_size: int,
                 epochs: int):
        self.image_size = image_size
        self.image_channel = image_channel
        self.image_patch_size = image_patch_size
        self.attention_head = attention_head
        self.attention_layer = attention_layer
        self.eps = eps
        self.dropout = dropout
        self.batch_size = batch_size
        self.hidden_layer = hidden_layer
        self.embedding_dim = image_channel * (image_patch_size ** 2)
        self.epochs = epochs
