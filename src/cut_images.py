import cv2

from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder

images = get_all_files_in_folder(Path('data/cut_images/input'), ['*.png'])

for path in tqdm(images):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = img[:512, :512]
    cv2.imwrite('data/cut_images/output/' + path.name, img)
