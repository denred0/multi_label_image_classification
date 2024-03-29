import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

import shutil
import cv2
from torch import nn
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

# classLabels = ["picture", "pushed", "wrinkle", "break_defect"]
classLabels = ["risunok", "nadav", "morshiny", "izlom"]

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

create_labels = False

if create_labels:
    images_list = []
    picture = [0] * len(images)
    pushed = [0] * len(images)
    wrinkle = [0] * len(images)
    break_defect = [0] * len(images)

    for i, image in enumerate(tqdm(images, desc='Copying images... ')):
        shutil.copy(str(image), train_images_dir)
        images_list.append(image.name)

        for mask in masks:
            if mask.stem == image.stem:
                m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

                classes = np.unique(m).tolist()
                for cl in classes:
                    cl_name = classLabels_dict.get(cl)

                    if cl_name == 'picture':
                        picture[i] = 1

                    if cl_name == 'pushed':
                        pushed[i] = 1

                    if cl_name == 'wrinkle':
                        wrinkle[i] = 1

                    if cl_name == 'break_defect':
                        break_defect[i] = 1

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df_train = pd.DataFrame(list(zip(images_list, picture, pushed, wrinkle, break_defect)),
                            columns=['image', 'picture', 'pushed', 'wrinkle', 'break_defect'])

    df_valid = pd.DataFrame(columns=df_train.columns)

    for index, row in df_train.iterrows():
        if index % 10 == 0:
            df_valid = df_valid.append(row, ignore_index=True)
            df_train.drop(index, inplace=True)

    df_train.to_csv('data/prepare_data/images_masks/output/data_train.csv', index=None)
    df_valid.to_csv('data/prepare_data/images_masks/output/data_valid.csv', index=None)
    print(df_train.head())


def visualization():
    df = pd.read_csv('data/prepare_data/images_masks/output/data.csv')
    fig1, ax1 = plt.subplots()
    df.iloc[:, 1:].sum(axis=0).plot.pie(autopct='%1.1f%%', shadow=True, startangle=90, ax=ax1)
    ax1.axis("equal")
    plt.show()

    sns.heatmap(df.iloc[:, 1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)
    plt.show()

    def visualizeImage(idx):
        fd = df.iloc[idx]
        image = fd.image
        label = fd[1:].tolist()
        print(image)
        image = Image.open(images_dir + '/' + image)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.grid(False)
        classes = np.array(classLabels)[np.array(label, dtype=np.bool)]
        for i, s in enumerate(classes):
            ax.text(0, i * 20, s, verticalalignment='top', color="white", fontsize=16, weight='bold')
        plt.show()

    visualizeImage(52)


# visualization()


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


model_type = 'resnet152d'

lr = 0.0003
batch_size = 8
epochs = 80
input_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

exp_name = model_type + '_ExponentialLR_new_aug_new_dataset_testset3_e80_' + 'batch_' + str(batch_size) + '_lr_' + str(lr)
if not os.path.exists('logs/' + exp_name):
    os.makedirs('logs/' + exp_name)

model = timm.create_model(model_type, pretrained=True)
conf = model.default_cfg

last_linear = conf['classifier']

# for param in model.parameters():
#     param.requires_grad_(False)

if last_linear == 'fc':
    num_features = model.fc.in_features
    top_head = create_head(num_features, len(classLabels))
    model.fc = top_head
elif last_linear == 'classifier':
    num_features = model.classifier.in_features
    top_head = create_head(num_features, len(classLabels))
    model.classifier = top_head
elif last_linear == 'last_linear':
    num_features = model.last_linear.in_features
    top_head = create_head(num_features, len(classLabels))
    model.last_linear = top_head
elif last_linear == 'head':
    num_features = model.head.in_features
    top_head = create_head(num_features, len(classLabels))
    model.head = top_head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=False)

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

transformsA = A.Compose([A.Resize(input_size, input_size),
                         A.Normalize(mean=mean, std=std),
                         ToTensorV2()])

dataset_train = MyDataset('data/train_data/data_train.csv', Path(images_dir), augments,
                          transformsA)
dataset_valid = MyDataset('data/train_data/data_valid.csv', Path(images_dir), None, transformsA)

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


def create_checkpoint(model, epoch, filename, exp_name, type):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "batch_size": batch_size,
    }  # save all important stuff

    if type == 'best':
        old_best = [chkp for chkp in os.listdir('logs/' + exp_name) if chkp.startswith("best_")]
        if len(old_best):
            os.remove('logs/' + exp_name + '/' + old_best[0])

    if type == 'last':
        old_last = [chkp for chkp in os.listdir('logs/' + exp_name) if chkp.startswith("last_")]
        if len(old_last):
            os.remove('logs/' + exp_name + '/' + old_last[0])

    torch.save(checkpoint, 'logs/' + exp_name + '/' + filename)


train_acc = []
val_acc = []
train_loss = []
val_loss = []


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5):
    min_loss = 1000
    max_acc = 0

    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'val']:
            if phase == "train":  # put the model in training mode
                model.train()
                # scheduler.step()
            else:  # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0

            for data, target in data_loader[phase]:
                # load the data and target to respective device
                data, target = data.to(device), target.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    # feed the input
                    output = model(data)
                    # calculate the loss
                    loss = criterion(output, target)
                    preds = torch.sigmoid(output).data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # update the model parameters
                        optimizer.step()

                        # if phase == "train":
                        #     scheduler.step()

                        # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += f1_score(target.to("cpu").to(torch.int).numpy(),
                                             preds.to("cpu").to(torch.int).numpy(), average="samples",
                                             zero_division=1) * data.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)

            result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "train":  # put the model in training mode
                scheduler.step()
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)

            # save checkpoint
            if phase == "val":
                print(result)

                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

                checkpoint_name = f'{exp_name}_{round(epoch_acc, 4)}_{round(epoch_loss, 4)}_e{epoch}.pt'
                if epoch_loss < min_loss or epoch_acc > max_acc:
                    create_checkpoint(model, epoch, 'best_' + checkpoint_name, exp_name, 'best')
                    min_loss = epoch_loss
                    max_acc = epoch_acc

                create_checkpoint(model, epoch, 'last_' + checkpoint_name, exp_name, 'last')


train(model, dataloader, criterion, optimizer, scheduler, num_epochs=epochs)

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# best_chkp = [chkp for chkp in os.listdir('logs/' + exp_name) if chkp.startswith("best_")]
# checkpoint = torch.load(Path('logs/' + exp_name + '/' + best_chkp[0]))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# batch_size = checkpoint['batch_size']
# #
# model.eval()  ## or model.train()
#
# image, label = next(iter(dataloader["val"]))
# image = image.to(device)
# label = label.to(device)
# output = 0
# with torch.no_grad():
#     output = model(image)
# output = torch.sigmoid(output)
#
# output = output > 0.3
#
# mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
#
#
# def denormalize(image):
#     image = image.to("cpu").clone().detach()
#     image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
#     image = image.permute(1, 2, 0)
#     image = torch.clamp(image, 0, 1)
#     return image.numpy()
#
#
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
#
#
# for i in range(batch_size):
#     visualize(image[i], label[i].tolist(), output[i].tolist())
