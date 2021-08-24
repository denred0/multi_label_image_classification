from warnings import simplefilter

simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UserWarning)

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

from sklearn.metrics import precision_score, recall_score, f1_score

import timm

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import trange
from sklearn.metrics import precision_score, f1_score

from dataset_wrapper import MyDataset

# classLabels = ["picture", "pushed", "wrinkle", "break_defect"]
classLabels = ["risunok", "nadav", "morshiny", "izlom"]
picture = []
pushed = []
wrinkle = []
break_defect = []

classLabels_dict = {1: "risunok", 2: "nadav", 3: "morshiny", 4: "izlom"}

images_dir = 'data/prepare_data/images_masks/output/images'

dirpath = Path('inference')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

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
dataset_valid = MyDataset('data/prepare_data/images_masks/output/data_valid.csv', Path(images_dir), None, transformsA)
dataloader = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)

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


mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

min_conf = 0.5
# counter = 0

conf_of_TP_sum = 0
precision_sum = 0
recall_sum = 0
f1_sum = 0

log_info = []

for i, (image, labels) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
    image = image.to(device)
    label = labels.to(device)
    output = 0
    with torch.no_grad():
        output = model(image)
    output = torch.sigmoid(output)
    output = output.cpu().detach().numpy()[0]

    image = (denormalize(image[0]) * 255).astype(int)

    image_name = dataloader.dataset.df['image'][i]

    cv2.imwrite('inference/' + image_name, image)
    image_draw = cv2.imread('inference/' + image_name, cv2.IMREAD_COLOR)

    # image_draw = image.copy()

    classes_gt = np.array(classLabels)[np.array(labels[0].tolist(), dtype=np.bool)]
    classes_gt_metrics = [int(x) for x in labels[0].tolist()]

    classes_pred = {}
    classes_pred_metrics = [0] * len(labels[0].tolist())
    conf_of_TP_list = []
    for j, conf in enumerate(output):
        if conf >= min_conf:
            classes_pred[classLabels[j]] = round(conf, 2)
            classes_pred_metrics[j] = 1

            if classes_pred_metrics[j] == classes_gt_metrics[j] == 1:
                conf_of_TP_list.append(round(conf, 2))

    # calc metrics
    precision = precision_score(classes_gt_metrics, classes_pred_metrics, zero_division=1)
    recall = recall_score(classes_gt_metrics, classes_pred_metrics)
    f1 = f1_score(classes_gt_metrics, classes_pred_metrics)

    conf_of_TP = 0
    if len(conf_of_TP_list) != 0:
        conf_of_TP = sum(conf_of_TP_list) / len(conf_of_TP_list)

    precision_sum += precision
    recall_sum += recall
    f1_sum += f1
    conf_of_TP_sum += conf_of_TP

    log_info.append(
        [image_name, classes_gt.tolist(), list(classes_pred.keys()), round(precision, 3), round(recall, 3),
         round(f1, 3),
         round(conf_of_TP, 3)])

    classes_pred = {k: v for k, v in sorted(classes_pred.items(), key=lambda item: item[1], reverse=True)}

    for pos, gt in enumerate(classes_gt):
        cv2.putText(image_draw, gt, (20, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for pos, (cl, pr) in enumerate(classes_pred.items()):
        cv2.putText(image_draw, str(cl) + ' ' + str(pr), (200, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

    cv2.imwrite('inference/' + image_name, image_draw)

precision_total = precision_sum / len(dataloader.dataset)
recall_total = recall_sum / len(dataloader.dataset)
f1_total = f1_sum / len(dataloader.dataset)
conf_of_TP_total = conf_of_TP_sum / len(dataloader.dataset)

print('precision_total', round(precision_total, 3))
print('recall_total', round(recall_total, 3))
print('f1_total', round(f1_total, 3))
print('conf_of_TP_total', round(conf_of_TP_total, 3))

with open(Path('inference').joinpath('log.txt'), 'w') as f:
    f.write("%s\n" % '[image_name, gt_classes, pred_classes, precision, recall, f1, conf_of_TP]')
    f.write(
        "%s\n" % f'[all_images, [], [], {round(precision_total, 3)}, {round(recall_total, 3)}, {round(f1_total, 3)}, {round(conf_of_TP_total, 3)}]')
    for item in log_info:
        f.write("%s\n" % item)
