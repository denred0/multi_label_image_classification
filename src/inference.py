from warnings import simplefilter

simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import torch

import shutil
import cv2
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import timm

import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import precision_score, f1_score

from dataset_wrapper import MyDataset

images_output_folder = Path('data/inference/inference_error')
if images_output_folder.exists() and images_output_folder.is_dir():
    shutil.rmtree(images_output_folder)
Path(images_output_folder).mkdir(parents=True, exist_ok=True)

images_output_folder = Path('data/inference/inference_valid')
if images_output_folder.exists() and images_output_folder.is_dir():
    shutil.rmtree(images_output_folder)
Path(images_output_folder).mkdir(parents=True, exist_ok=True)

# classLabels = ["picture", "pushed", "wrinkle", "break_defect"]
classLabels = ["risunok", "nadav", "morshiny", "izlom"]
picture = []
pushed = []
wrinkle = []
break_defect = []

classLabels_dict = {1: "risunok", 2: "nadav", 3: "morshiny", 4: "izlom"}

images_dir = 'data/prepare_data/images_masks/output/images'

transformsA = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         ToTensorV2()])

batch_size = 1
dataset_valid = MyDataset('data/train_data/data_test.csv', Path(images_dir), None, transformsA)
dataloader = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)

model_type = 'resnet152d'
model = timm.create_model(model_type, pretrained=True)
num_features = model.fc.in_features


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


top_head = create_head(num_features, len(classLabels))
model.fc = top_head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=False)

# best_chkp = [chkp for chkp in os.listdir('logs/' + exp_name) if chkp.startswith("best_")]
checkpoint = torch.load(
    Path(
        'logs/resnet152d_ExponentialLR_new_aug_new_dataset_testset3_e80_batch_8_lr_0.0003/best_resnet152d_ExponentialLR_new_aug_new_dataset_testset3_e80_batch_8_lr_0.0003_0.9699_0.0888_e71.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
batch_size = checkpoint['batch_size']
model.eval()


def denormalize(image):
    image = image.to("cpu").clone().detach()
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)
    image = torch.clamp(image, 0, 1)
    return image.numpy()


mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

min_conf = 0.35
# counter = 0

conf_of_TP_list = []
precision_list = []
recall_list = []
f1_list = []
accuracy_list = []
images_names = []

accuracy_classes = [[] for _ in classLabels_dict.keys()]

# for _ in classLabels_dict.keys():
#     precision_classes.append([])


link_start_number = 103035
folder_link = 'https://nxc.videomatrix.ru:8899/s/rRL3tWdGrE9JwC2'
link = 'https://nxc.videomatrix.ru:8899/s/rRL3tWdGrE9JwC2?dir=undefined&openfile='
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
    accuracy = accuracy_score(classes_gt_metrics, classes_pred_metrics)
    f1 = f1_score(classes_gt_metrics, classes_pred_metrics)

    for v in range(len(classes_gt_metrics)):
        accuracy_classes[v].append(1 if classes_gt_metrics[v] == classes_pred_metrics[v] else 0)

    conf_of_TP = 0
    if len(conf_of_TP_l) != 0:
        conf_of_TP = sum(conf_of_TP_l) / len(conf_of_TP_l)

    precision_list.append(round(precision * 100, 1))
    recall_list.append(round(recall * 100, 1))
    f1_list.append(round(f1 * 100, 1))
    accuracy_list.append(round(accuracy * 100, 1))
    conf_of_TP_list.append(round(conf_of_TP * 100, 1))

    # draw results on image
    classes_pred = {k: v for k, v in sorted(classes_pred.items(), key=lambda item: item[1], reverse=True)}

    for pos, gt in enumerate(classes_gt):
        cv2.putText(image_draw, gt, (20, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for pos, (cl, pr) in enumerate(classes_pred.items()):
        cv2.putText(image_draw, str(cl) + ' ' + str(pr), (200, (pos + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

    if f1 == 1:
        cv2.imwrite('data/inference/inference_valid/' + image_name, image_draw)
    else:
        cv2.imwrite('data/inference/inference_error/' + image_name, image_draw)

    links.append(link + str(link_start_number))
    link_start_number += 1

precision_total = sum(precision_list) / len(precision_list)
recall_total = sum(recall_list) / len(recall_list)
f1_total = sum(f1_list) / len(f1_list)
conf_of_TP_total = sum(conf_of_TP_list) / len(conf_of_TP_list)
accuracy_total = sum(accuracy_list) / len(accuracy_list)

accuarcy_total_classes = []
for a in accuracy_classes:
    accuarcy_total_classes.append(sum(a) / len(a))

print('Accuracy:')
for i, val in enumerate(classLabels_dict.values()):
    print(f'{val}: {round(accuarcy_total_classes[i] * 100, 3)}')
print(f'Mean: {round(np.mean(accuarcy_total_classes) * 100, 3)}')

print('\nprecision_total', round(precision_total, 2))
print('recall_total', round(recall_total, 3))
print('f1_total', round(f1_total, 3))
print('accuracy', round(accuracy_total, 3))
print('conf_of_TP_total', round(conf_of_TP_total, 3))

with open(Path('data/inference').joinpath('Общие_результаты.txt'), 'w') as f:
    f.write("%s\n" % f'Количество изображений: {len(precision_list)}')
    f.write("%s\n" % f'Общая точность (precision): {round(precision_total, 1)}%')
    f.write("%s\n" % f'Общая полнота (recall): {round(precision_total, 1)}%')
    f.write("%s\n" % f'F1-мера (F1): {round(f1_total, 1)}%')
    f.write("%s\n" % f'Уверенность правильно найденных классов (TP): {round(conf_of_TP_total, 1)}%')
    f.write("%s\n" % f'Пороговая уверенность: {min_conf * 100}%')
    f.write("%s\n" % '')

    f.write("%s\n" % f'Точность по классам, %:')
    for i, val in enumerate(classLabels_dict.values()):
        f.write("%s\n" % f'{val}: {round(accuarcy_total_classes[i] * 100, 3)}')

    f.write("%s\n" % f'Папка с изображениями: {folder_link}')

augments = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.3),
    A.Flip(p=0.5),
    A.CLAHE(clip_limit=4, p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Emboss(p=0.5),
    A.Sharpen(p=0.5),
    A.GridDistortion(p=0.5),
    A.ImageCompression(quality_lower=85, p=0.5),
    A.Superpixels(p=0.5)

    # A.GaussNoise(p=0.4),
    # A.OneOf([A.MotionBlur(p=0.5),
    #          A.MedianBlur(blur_limit=3, p=0.5),
    #          A.Blur(blur_limit=3, p=0.1)], p=0.5),
    # A.OneOf([A.CLAHE(clip_limit=4),
    #          A.Sharpen(),
    #          A.Emboss(),
    #          A.RandomBrightnessContrast()], p=0.5)
], p=0.7)

model = 'resnet152d'
lr = 0.0003
batch_size = 8
epochs = 80
input_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

optimizer = 'optim.Adam(model.parameters(), lr=lr)'
scheduler = 'lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=False)'
scheduler_strategy = 'epoch'

with open(Path('data/inference').joinpath('Спецификация решения.txt'), 'w') as f:
    f.write("%s\n" % f'Augmentations:')
    f.write("%s\n" % f'{augments}')
    f.write("%s\n" % f'Model: {model}')
    f.write("%s\n" % f'lr: {lr}')
    f.write("%s\n" % f'batch_size: {batch_size}')
    f.write("%s\n" % f'epochs: {epochs}')
    f.write("%s\n" % f'input_size: {input_size}')
    f.write("%s\n" % f'mean: {mean}')
    f.write("%s\n" % f'std: {std}')
    f.write("%s\n" % f'optimizer: {optimizer}')
    f.write("%s\n" % f'scheduler: {scheduler}')
    f.write("%s\n" % f'scheduler_strategy: {scheduler_strategy}')

df = pd.DataFrame(list(zip(images_names, precision_list, recall_list, f1_list, conf_of_TP_list, links)),
                  columns=['Изображение', 'Точность', 'Полнота', 'F1-мера', 'Уверенность правильно найденных классов',
                           'Ссылка на изображение'])
df.to_csv('data/inference/Изображения_результаты.csv', index=False)
