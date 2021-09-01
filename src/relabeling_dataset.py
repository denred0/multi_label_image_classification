import shutil
from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder

old_images = get_all_files_in_folder(Path('data/relabeling_dataset/old'), ['*.png'])
new_images = get_all_files_in_folder(Path('data/relabeling_dataset/new'), ['*.png'])
new_images_names = [x.stem for x in new_images]

dirpath = Path('data/relabeling_dataset/result')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

for image in tqdm(old_images):
    if image.stem not in new_images_names:
        shutil.copy(image, 'data/relabeling_dataset/result')
        shutil.copy('data/relabeling_dataset/old/' + image.stem + '.txt', 'data/relabeling_dataset/result')

for image in tqdm(new_images):
    shutil.copy(image, 'data/relabeling_dataset/result')
    shutil.copy('data/relabeling_dataset/new/' + image.stem + '.txt', 'data/relabeling_dataset/result')