import os
import shutil
import pandas as pd

from pathlib import Path


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    # files_grabbed = [file[:-4] for file in files_grabbed if 'orig' in file.lower()]
    return files_grabbed


dir = 'data/create_labels/input'

all_images = []
all_txts = []

for subdir, dirs, files in os.walk(dir):
    for folder in dirs:
        p = os.path.join(dir, folder) + os.path.sep
        images = get_all_files_in_folder(Path(p), ['*.png'])
        all_images.extend(images)

        txts = get_all_files_in_folder(Path(p), ['*.txt'])
        all_txts.extend(txts)

images_list = []
picture = []
pushed = []
wrinkle = []
break_defect = []

images_output_folder = Path('data/create_labels/output/images')
if images_output_folder.exists() and images_output_folder.is_dir():
    shutil.rmtree(images_output_folder)
Path(images_output_folder).mkdir(parents=True, exist_ok=True)

for img in all_images:
    images_list.append(img.name)

    shutil.copy(img, images_output_folder)

    for txt in all_txts:
        if txt.stem == img.stem:
            with open(str(txt)) as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]

            defects = lines[1].split(' ')
            if defects[0] in ['1', '8']:
                picture.append(1)
            else:
                picture.append(0)

            if defects[1] in ['1', '8']:
                pushed.append(1)
            else:
                pushed.append(0)

            if defects[2] in ['1', '8']:
                wrinkle.append(1)
            else:
                wrinkle.append(0)

            if defects[3] in ['1', '8']:
                break_defect.append(1)
            else:
                break_defect.append(0)

df_train = pd.DataFrame(list(zip(images_list, picture, pushed, wrinkle, break_defect)),
                        columns=['image', 'picture', 'pushed', 'wrinkle', 'break_defect'])

df_valid_test = pd.DataFrame(columns=df_train.columns)

for index, row in df_train.iterrows():
    if index % 5 == 0:
        df_valid_test = df_valid_test.append(row, ignore_index=True)
        df_train.drop(index, inplace=True)

df_valid = df_valid_test.iloc[:len(df_valid_test) // 2]
df_test = df_valid_test.iloc[len(df_valid_test) // 2:]

df_train.to_csv('data/create_labels/output/data_train.csv', index=None)
df_valid.to_csv('data/create_labels/output/data_valid.csv', index=None)
df_test.to_csv('data/create_labels/output/data_test.csv', index=None)



print()
