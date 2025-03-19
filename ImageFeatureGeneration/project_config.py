class ProjectConfiguration:
    def __init__(self,
                 image_size: int = 128,
                 image_channel: int = 3,
                 blob_size: int = 32):
        self.image_size = image_size
        self.image_channel = image_channel
        self.blob_size = blob_size
