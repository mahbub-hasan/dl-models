class ProjectConfiguration:
    def __init__(self,
                 image_size: int = 224,
                 image_channel: int = 3,
                 patch_size: int = 16,
                 attention_head: int = 4,
                 attention_layer: int = 12,
                 batch_size: int = 32,
                 eps: float = 1e-6,
                 hidden_layer: int = 2048,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 **kwargs):
        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.attention_head = attention_head
        self.attention_layer = attention_layer
        self.batch_size = batch_size
        self.eps = eps
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.image_embedding = image_channel * (patch_size**2)
