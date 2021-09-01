from pathlib import Path
from tqdm import tqdm

from my_utils import get_all_files_in_folder

txts = get_all_files_in_folder(Path('data/dataset_versions/dataset_ver2/all_images'), ['*.txt'])

classLabels = ["risunok", "nadav", "morshiny", "izlom"]

classes_count = {}

for txt in tqdm(txts):
    with open(txt) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines[1:]]

    for i, l in enumerate(lines[0].split(' ')):
        if int(l) == 1:
            classes_count[classLabels[i]] = classes_count.get(classLabels[i], 0) + 1

for k, v in classes_count.items():
    print(f'{k}: {v}, {round(v / sum(classes_count.values()) * 100, 1)}%')

print(classes_count)
