import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, augments=None, preprocessing=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.augments = augments
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = cv2.imread(str(self.img_dir / d.image), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)

        if self.augments:
            augmented = self.augments(image=image)
            image = augmented['image']

        if self.preprocessing:
            preprocessed = self.preprocessing(image=image)
            image = preprocessed['image']

        # if self.transforms is not None:
        #     image = self.transforms(image)

        return image, label
