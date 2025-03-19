import os
import pandas as pd
import numpy as np
from project_config import ProjectConfiguration
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms


class LoadFile:
    def __init__(self,
                 config: ProjectConfiguration,
                 csv_file_path: str,
                 image_file_path: str):
        self.config = config
        self.csv_file_path = csv_file_path
        self.image_file_path = image_file_path
        self.img_transform = transforms.Compose([
            transforms.Resize(size=(config.image_size, config.image_size)),
        ])
        self.edge_blob = 32
        self.step_size = 32

    def preprocessing_file(self):
        df = pd.read_csv(self.csv_file_path, sep=",")

        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['dx'])

        print(encoder.classes_)
        print(encoder.transform(encoder.classes_))

        sample_size = df['target'].value_counts().min()
        print(f"Taking sample size: {sample_size}")

        df_of_akiec = df[df['target'] == 0][:sample_size]
        df_of_bcc = df[df['target'] == 1][:sample_size]
        df_of_bkl = df[df['target'] == 2][:sample_size]
        df_of_df = df[df['target'] == 3][:sample_size]
        df_of_mel = df[df['target'] == 4][:sample_size]
        df_of_nv = df[df['target'] == 5][:sample_size]
        df_of_vasc = df[df['target'] == 6][:sample_size]

        new_df = pd.concat([df_of_akiec, df_of_bcc, df_of_bkl, df_of_df, df_of_mel, df_of_nv, df_of_vasc], axis=0)

        image_path = [os.path.join(self.image_file_path, image_name) for image_name in new_df['image_id']]

        new_df['image_path'] = image_path

        df_final = new_df[['image_id', 'image_path', 'target']]

        return df_final

    def make_feature_dataset(self):
        df = self.preprocessing_file()

        print(f"Final dataset shape: {df.shape}")

        image_features_vector = np.zeros(shape=(df.shape[0], 120))
        image_target = np.zeros(shape=(df.shape[0], 1))
        image_name = [None] * df.shape[0]
        count_blob = 0
        for idx, (image_path, target, image_name_id) in enumerate(zip(df['image_path'], df['target'], df['image_id'])):
            img_path = image_path + '.jpg'
            image_name[idx] = image_name_id
            img = Image.open(img_path).convert('RGB')
            # change image shape --> [128, 128]
            img = self.img_transform(img)

            row, col = img.size
            img_array = np.array(img)

            feature = np.zeros(shape=(120, 1))

            # pixels_row = row - 2 * self.edge_blob
            # # NBE is the number of blobs for each image
            # NBE = ((pixels_row - self.edge_blob) // self.step_size + 1) * (
            #         (pixels_row - self.edge_blob) // self.step_size + 1)
            #
            # # x is the matrix who's the rows contain the feature vectors

            count = 0
            for i in range(self.edge_blob, (row - 2 * self.edge_blob + 2), self.step_size):
                for j in range(self.edge_blob, (col - 2 * self.edge_blob + 2), self.step_size):
                    for channel in range(len(img.getbands())):
                        # average of the central blob
                        M = []
                        for k in range(i, i + self.edge_blob - 1):
                            M.extend(img_array[k, j:j + self.edge_blob - 1, channel])
                        B1 = np.mean(M)
                        B2 = np.var(M)

                        feature[(10 * count)] = B1
                        feature[(10 * count + 1)] = B2

                        # average of the right blob
                        M = []
                        for k in range(i, i + self.edge_blob - 1):
                            M.extend(img_array[k, j + self.edge_blob:j + 2 * self.edge_blob - 1, channel])
                        B1 = np.mean(M)
                        B2 = np.var(M)

                        feature[(10 * count + 2)] = feature[(10 * count)] - B1
                        feature[(10 * count + 3)] = feature[(10 * count + 1)] - B2

                        # average of the left blob
                        M = []
                        for k in range(i, i + self.edge_blob - 1):
                            M.extend(img_array[k, j - self.edge_blob:j - 1, channel])
                        B1 = np.mean(M)
                        B2 = np.var(M)

                        feature[(10 * count + 4)] = feature[(10 * count)] - B1
                        feature[(10 * count + 5)] = feature[(10 * count + 1)] - B2

                        # average of the upper blob
                        M = []
                        for k in range(i - self.edge_blob, i - 1):
                            M.extend(img_array[k, j:j + self.edge_blob - 1, channel])
                        B1 = np.mean(M)
                        B2 = np.var(M)

                        feature[(10 * count + 6)] = feature[(10 * count)] - B1
                        feature[(10 * count + 7)] = feature[(10 * count + 1)] - B2

                        # average of the lower blob

                        M = []
                        for k in range(i + self.edge_blob, i + 2 * self.edge_blob - 1):
                            M.extend(img_array[k, j:j + self.edge_blob - 1, channel])
                        B1 = np.mean(M)
                        B2 = np.var(M)

                        feature[(10 * count + 8)] = feature[(10 * count)] - B1
                        feature[(10 * count + 9)] = feature[(10 * count + 1)] - B2

                        count += 1
            image_features_vector[count_blob, :] = feature.T
            image_target[count_blob] = target
            count_blob += 1
        return image_name, image_features_vector, image_target

    def getDataset(self):
        image_name, image_features, image_target = self.make_feature_dataset()
        df_image_name = pd.DataFrame(image_name, columns=['image_name'])
        df_image_features = pd.DataFrame(image_features, columns=[f'feature_{i+1}' for i in range(image_features.shape[1])])
        df_image_target = pd.DataFrame(image_target, columns=['target'])

        df = pd.concat([df_image_name, df_image_features, df_image_target], axis=1)

        print(df.shape)

        df.to_csv('melanoma_feature_dataset.csv', index=False)

        print("-"*30)
        print("Dataset Create Successfully")
        print("-"*30)

