import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

import shutil
import cv2
# from scipy.stats import triang_gen
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import json
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from my_utils import get_all_files_in_folder
from tqdm import tqdm
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import trange
from sklearn.metrics import precision_score, f1_score

from dataset_wrapper import MyDataset

classLabels = ["picture", "pushed", "wrinkle", "break_defect"]

picture = []
pushed = []
wrinkle = []
break_defect = []

classLabels_dict = {1: "picture", 2: "pushed", 3: "wrinkle", 4: "break_defect"}

image_ext = 'png'

images_dir = 'data/prepare_data/images_masks/output/images'

train_images_dir = 'data/prepare_data/images_masks/output/images'
if not os.path.exists(train_images_dir):
    os.makedirs(train_images_dir)

images = get_all_files_in_folder(Path('data/prepare_data/images_masks/input/images'), ['*.' + image_ext])
masks = get_all_files_in_folder(Path('data/prepare_data/images_masks/input/masks'), ['*.' + image_ext])

augments = A.Compose([
    # A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.3),
    # A.GaussNoise(p=0.4),
    # A.OneOf([A.MotionBlur(p=0.5),
    #          A.MedianBlur(blur_limit=3, p=0.5),
    #          A.Blur(blur_limit=3, p=0.1)], p=0.5),
    # A.OneOf([A.CLAHE(clip_limit=2),
    #          A.Sharpen(),
    #          A.Emboss(),
    #          A.RandomBrightnessContrast()], p=0.5)
])

transformsA = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         ToTensorV2()])

batch_size = 1

dataset_train = MyDataset('data/prepare_data/images_masks/output/data_train.csv', Path(images_dir), augments,
                          transformsA)
dataset_valid = MyDataset('data/prepare_data/images_masks/output/data_valid.csv', Path(images_dir), None, transformsA)

print(f"trainset len {len(dataset_train)} valset len {len(dataset_valid)}")
dataloader = {"train": DataLoader(dataset_train, shuffle=True, batch_size=batch_size),
              "val": DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)}

# different networks
# sheduller step = epoch
# different Shedullerrs
# loss function
# change initial lr


# model_type = 'tf_efficientnet_b6_ns'
# model = timm.create_model(model_type, pretrained=True)
# num_features = model.classifier.in_features
# model.classifier = nn.Linear(num_features, len(classLabels_dict.keys()))

model_type = 'resnet152d'
model = timm.create_model(model_type, pretrained=True)

num_features = model.fc.in_features
# model.classifier = nn.Linear(num_features, len(classLabels_dict.keys()))

# model = models.resnet152(pretrained=True)  # load the pretrained model
# num_features = model.fc.in_features  # get the no of on_features in last Linear unit
# print(num_features)


# freeze the entire convolution base
for param in model.parameters():
    param.requires_grad_(False)


def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features // 2, num_features // 4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0:
            layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)


top_head = create_head(num_features, len(classLabels))  # because ten classes
# model.fc = top_head  # replace the fully connected layer
model.fc = top_head  # replace the fully connected layer
# model.last_linear = top_head  # replace the fully connected layer

# print(model)

exp_name = model_type + '_aug_ExponentialLR'
if not os.path.exists('logs/' + exp_name):
    os.makedirs('logs/' + exp_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=False)

best_chkp = [chkp for chkp in os.listdir('logs/' + exp_name) if chkp.startswith("best_")]
checkpoint = torch.load(Path('logs/resnet152d_ExponentialLR/best_resnet152d_ExponentialLR_0.8716_0.2878_e10.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
batch_size = checkpoint['batch_size']
#
model.eval()  ## or model.train()


def denormalize(image):
    image = image.to("cpu").clone().detach()
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)
    image = torch.clamp(image, 0, 1)
    return image.numpy()


# image, label = next(iter(dataloader["val"]))
# image = image.to(device)
# label = label.to(device)
# output = 0
# with torch.no_grad():
#     output = model(image)
# output = torch.sigmoid(output)

# output = output > 0.3

mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

min_conf = 0.3
counter = 0

for image, labels in (dataloader["val"]):
    image = image.to(device)
    label = labels.to(device)
    output = 0
    with torch.no_grad():
        output = model(image)
    output = torch.sigmoid(output)
    output = output.cpu().detach().numpy()[0]

    image = (denormalize(image[0]) * 255).astype(int)
    cv2.imwrite('inference/' + str(counter) + '.png', image)
    image_draw = cv2.imread('inference/' + str(counter) + '.png', cv2.IMREAD_COLOR)

    # image_draw = image.copy()

    classes_gt = np.array(classLabels)[np.array(labels[0].tolist(), dtype=np.bool)]

    classes_pred = {}
    for i, conf in enumerate(output):
        if conf >= min_conf:
            classes_pred[classLabels[i]] = round(conf, 2)

    classes_pred = {k: v for k, v in sorted(classes_pred.items(), key=lambda item: item[1], reverse=True)}

    for pos, gt in enumerate(classes_gt):
        cv2.putText(image_draw, gt, (20, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for pos, (cl, pr) in enumerate(classes_pred.items()):
        cv2.putText(image_draw, str(cl) + ' ' + str(pr), (300, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite('inference/' + str(counter) + '.png', image_draw)

    counter += 1


# def visualize(image, actual, pred):
#     fig, ax = plt.subplots()
#     ax.imshow(denormalize(image))
#     ax.grid(False)
#     classes = np.array(classLabels)[np.array(actual, dtype=np.bool)]
#     for i, s in enumerate(classes):
#         ax.text(0, i * 20, s, verticalalignment='top', color="green", fontsize=16, weight='bold')
#
#     classes = np.array(classLabels)[np.array(pred, dtype=np.bool)]
#     for i, s in enumerate(classes):
#         ax.text(360, i * 20, s, verticalalignment='top', color="red", fontsize=16, weight='bold')
#
#     plt.show()
#     plt.savefig('inference/foo.png')
#
#
# for i in range(batch_size):
#     visualize(image[i], label[i].tolist(), output[i].tolist())
