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

conf_of_TP_list = []
precision_list = []
recall_list = []
f1_list = []
images_names = []

link_start_number = 99325
folder_link = 'https://nxc.videomatrix.ru:8899/s/iB7PwygMbiQgPgq'
link = 'https://nxc.videomatrix.ru:8899/s/iB7PwygMbiQgPgq?dir=undefined&openfile='
links = []

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
    images_names.append(image_name)

    # resave image because of strange OpenCV error
    cv2.imwrite('inference/' + image_name, image)
    image_draw = cv2.imread('inference/' + image_name, cv2.IMREAD_COLOR)

    classes_gt = np.array(classLabels)[np.array(labels[0].tolist(), dtype=np.bool)]
    classes_gt_metrics = [int(x) for x in labels[0].tolist()]

    classes_pred = {}
    classes_pred_metrics = [0] * len(labels[0].tolist())
    conf_of_TP_l = []
    for j, conf in enumerate(output):
        if conf >= min_conf:
            classes_pred[classLabels[j]] = round(conf, 2)
            classes_pred_metrics[j] = 1

            # calc confidence of TP
            if classes_pred_metrics[j] == classes_gt_metrics[j] == 1:
                conf_of_TP_l.append(round(conf, 3))

    # calc metrics
    precision = precision_score(classes_gt_metrics, classes_pred_metrics, zero_division=1)
    recall = recall_score(classes_gt_metrics, classes_pred_metrics)
    f1 = f1_score(classes_gt_metrics, classes_pred_metrics)

    conf_of_TP = 0
    if len(conf_of_TP_l) != 0:
        conf_of_TP = sum(conf_of_TP_l) / len(conf_of_TP_l)

    precision_list.append(round(precision * 100, 1))
    recall_list.append(round(recall * 100, 1))
    f1_list.append(round(f1 * 100, 1))
    conf_of_TP_list.append(round(conf_of_TP * 100, 1))

    # draw results on image
    classes_pred = {k: v for k, v in sorted(classes_pred.items(), key=lambda item: item[1], reverse=True)}

    for pos, gt in enumerate(classes_gt):
        cv2.putText(image_draw, gt, (20, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for pos, (cl, pr) in enumerate(classes_pred.items()):
        cv2.putText(image_draw, str(cl) + ' ' + str(pr), (200, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

    cv2.imwrite('inference/' + image_name, image_draw)

    links.append(link + str(link_start_number))
    link_start_number += 1

precision_total = sum(precision_list) / len(precision_list)
recall_total = sum(recall_list) / len(recall_list)
f1_total = sum(f1_list) / len(f1_list)
conf_of_TP_total = sum(conf_of_TP_list) / len(conf_of_TP_list)

print('precision_total', round(precision_total, 3))
print('recall_total', round(recall_total, 3))
print('f1_total', round(f1_total, 3))
print('conf_of_TP_total', round(conf_of_TP_total, 3))

with open(Path('inference').joinpath('Общие_результаты.txt'), 'w') as f:
    f.write("%s\n" % f'Количество изображений: {len(precision_list)}')
    f.write("%s\n" % f'Общая точность (precision): {round(precision_total, 1)}%')
    f.write("%s\n" % f'Общая полнота (recall): {round(precision_total, 1)}%')
    f.write("%s\n" % f'F1-мера (F1): {round(f1_total, 1)}%')
    f.write("%s\n" % f'Уверенность правильно найденных классов (TP): {round(conf_of_TP_total, 1)}%')
    f.write("%s\n" % f'Пороговая уверенность: {min_conf * 100}%')
    f.write("%s\n" % '')
    f.write("%s\n" % f'Папка с изображениями: {folder_link}')

df = pd.DataFrame(list(zip(images_names, precision_list, recall_list, f1_list, conf_of_TP_list, links)),
                  columns=['Изображение', 'Точность', 'Полнота', 'F1-мера', 'Уверенность правильно найденных классов',
                           'Ссылка на изображение'])
df.to_csv('inference/Изображения_результаты.csv', index=False)
